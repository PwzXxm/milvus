// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//	http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package storage

import (
	"bytes"
	"encoding/binary"
	"hash/crc32"
	"io"

	"github.com/cockroachdb/errors"
	"github.com/klauspost/compress/zstd"
)

// TODO: this is file format for LOB data for storage v1, maybe not needed

const (
	compressionMethodNone = iota
	compressionMethodZstd
)

const (
	MagicTrailerBytesSize = 8
)

var MagicTrailer = [MagicTrailerBytesSize]byte{'L', 'O', 'B', 'D', 'A', 'T', 'A', 'F'}

// CompressionMethod defines an interface for compression implementations
type CompressionMethod interface {
	Compress([]byte) ([]byte, error)
	Dictionary() []byte
	Id() uint8
}

// CompressionNone implementation
type compressionNone struct{}

func (c *compressionNone) Compress(data []byte) ([]byte, error) {
	return data, nil
}

func (c *compressionNone) Dictionary() []byte { return nil }

func (c *compressionNone) Id() uint8 { return compressionMethodNone }

// CompressionZstd implementation.
type compressionZstd struct {
	encoder    *zstd.Encoder
	dictionary []byte
}

func newCompressionZstd(samples []byte) (*compressionZstd, error) {
	dict, err := zstd.BuildDict(zstd.BuildDictOptions{
		Contents: [][]byte{samples},
	})
	if err != nil {
		return nil, err
	}
	enc, err := zstd.NewWriter(nil,
		zstd.WithEncoderDict(dict),
		zstd.WithEncoderLevel(zstd.SpeedBetterCompression))
	if err != nil {
		return nil, err
	}
	return &compressionZstd{encoder: enc, dictionary: dict}, nil
}

func (c *compressionZstd) Compress(data []byte) ([]byte, error) {
	return c.encoder.EncodeAll(data, nil), nil
}

func (c *compressionZstd) Dictionary() []byte { return c.dictionary }

func (c *compressionZstd) Id() uint8 { return compressionMethodZstd }

// ChunkHeader describes a chunk written to disk.
type ChunkHeader struct {
	RowCount       uint32
	UncompressedSz uint64
	CompressedSz   uint64
	CRC32          uint32
}

// ChunkIndexEntry now includes the row index (offsets within the chunk).
type ChunkIndexEntry struct {
	Offset     uint64   // file offset of the chunk (including header)
	Size       uint64   // total size of the chunk (header + compressed data)
	RowCount   uint32   // number of rows in the chunk
	RowOffsets []uint32 // slice of offsets (within the uncompressed chunk) for each row
}

// Trailer remains mostly unchanged except for how we serialize the chunk index entries.
type Trailer struct {
	ChunkIndices []ChunkIndexEntry
	ChunkCount   uint32
	DictOffset   uint64
	DictSize     uint32
	Compression  uint8
	TrailerSize  uint64
	Version      uint32
	Magic        [MagicTrailerBytesSize]byte
}

type LobWriter struct {
	writer         io.WriteSeeker
	compressor     CompressionMethod
	chunkBuffer    bytes.Buffer
	indices        []ChunkIndexEntry
	threshold      int
	rowCount       uint32
	sampleBuffer   bytes.Buffer
	maxSampleSz    int
	currentOffsets []uint32 // row offsets within the current chunk
}

func NewLobWriter(writer io.WriteSeeker, threshold int, maxSampleSz int, compressor CompressionMethod) *LobWriter {
	return &LobWriter{
		writer:      writer,
		threshold:   threshold,
		maxSampleSz: maxSampleSz,
		compressor:  compressor,
	}
}

// AddRow now records the current offset in the chunk buffer before writing the row.
func (lw *LobWriter) AddRow(row []byte) error {
	// Record the starting offset of this row in the uncompressed chunk.
	currentOffset := uint32(lw.chunkBuffer.Len())
	lw.currentOffsets = append(lw.currentOffsets, currentOffset)

	// Sample buffering remains unchanged.
	if lw.sampleBuffer.Len() < lw.maxSampleSz {
		remainingSize := lw.maxSampleSz - lw.sampleBuffer.Len()
		if len(row) > remainingSize {
			row = row[:remainingSize]
		}
		lw.sampleBuffer.Write(row)
	}

	// Write the length and then the row data.
	if err := binary.Write(&lw.chunkBuffer, binary.LittleEndian, uint32(len(row))); err != nil {
		return err
	}
	_, err := lw.chunkBuffer.Write(row)
	if err != nil {
		return err
	}

	lw.rowCount++
	if lw.chunkBuffer.Len() >= lw.threshold {
		return lw.flushChunk()
	}
	return nil
}

// flushChunk compresses and writes the current chunk, then records its metadata (including row offsets)
// in the chunk index entry.
func (lw *LobWriter) flushChunk() error {
	if lw.compressor == nil {
		return errors.New("compressor is not initialized")
	}

	if lw.chunkBuffer.Len() == 0 {
		return nil
	}

	uncompressed := lw.chunkBuffer.Bytes()
	compressed, err := lw.compressor.Compress(uncompressed)
	if err != nil {
		return err
	}
	crc := crc32.ChecksumIEEE(compressed)

	// Get current file offset.
	offset, _ := lw.writer.Seek(0, io.SeekCurrent)

	// Write the chunk header.
	chunkHeader := ChunkHeader{
		RowCount:       lw.rowCount,
		UncompressedSz: uint64(len(uncompressed)),
		CompressedSz:   uint64(len(compressed)),
		CRC32:          crc,
	}
	if err := binary.Write(lw.writer, binary.LittleEndian, chunkHeader); err != nil {
		return err
	}

	// Write the compressed chunk data.
	_, err = lw.writer.Write(compressed)
	if err != nil {
		return err
	}

	// Create and store the chunk index entry with row offsets.
	entry := ChunkIndexEntry{
		Offset:     uint64(offset),
		Size:       uint64(24 /* header size */ + len(compressed)),
		RowCount:   lw.rowCount,
		RowOffsets: lw.currentOffsets,
	}
	lw.indices = append(lw.indices, entry)

	// Reset for the next chunk.
	lw.chunkBuffer.Reset()
	lw.currentOffsets = nil
	lw.rowCount = 0
	return nil
}

// Close flushes the final chunk, writes the compressor dictionary, and then writes the trailer which now
// includes the complete chunk index with row offsets.
func (lw *LobWriter) Close() error {
	if err := lw.flushChunk(); err != nil {
		return err
	}

	// Write compressor dictionary.
	dictOffset, _ := lw.writer.Seek(0, io.SeekCurrent)
	dict := lw.compressor.Dictionary()
	if _, err := lw.writer.Write(dict); err != nil {
		return err
	}

	// Build trailer.
	trailerBuf := new(bytes.Buffer)
	// Write the number of chunks.
	if err := binary.Write(trailerBuf, binary.LittleEndian, uint32(len(lw.indices))); err != nil {
		return err
	}

	// Write each ChunkIndexEntry.
	for _, idx := range lw.indices {
		// Write fixed fields: Offset, Size, and RowCount.
		if err := binary.Write(trailerBuf, binary.LittleEndian, idx.Offset); err != nil {
			return err
		}
		if err := binary.Write(trailerBuf, binary.LittleEndian, idx.Size); err != nil {
			return err
		}
		if err := binary.Write(trailerBuf, binary.LittleEndian, idx.RowCount); err != nil {
			return err
		}
		// Write the number of row offsets.
		rowOffsetsCount := uint32(len(idx.RowOffsets))
		if err := binary.Write(trailerBuf, binary.LittleEndian, rowOffsetsCount); err != nil {
			return err
		}
		// Write each row offset.
		for _, offset := range idx.RowOffsets {
			if err := binary.Write(trailerBuf, binary.LittleEndian, offset); err != nil {
				return err
			}
		}
	}

	// Write dictionary information and other trailer fields.
	if err := binary.Write(trailerBuf, binary.LittleEndian, uint64(dictOffset)); err != nil {
		return err
	}
	if err := binary.Write(trailerBuf, binary.LittleEndian, uint32(len(dict))); err != nil {
		return err
	}
	if err := binary.Write(trailerBuf, binary.LittleEndian, lw.compressor.Id()); err != nil {
		return err
	}
	// For example, write a version field.
	if err := binary.Write(trailerBuf, binary.LittleEndian, uint32(1)); err != nil {
		return err
	}
	// Trailer size includes itself; here we add 12 bytes for the trailer header.
	trailerSize := uint64(trailerBuf.Len() + 12)
	if err := binary.Write(trailerBuf, binary.LittleEndian, trailerSize); err != nil {
		return err
	}
	// Write the magic bytes.
	if err := binary.Write(trailerBuf, binary.LittleEndian, MagicTrailer); err != nil {
		return err
	}

	_, err := lw.writer.Write(trailerBuf.Bytes())
	return err
}

type LobReader struct {
	reader     io.ReaderAt
	trailer    Trailer
	decoder    *zstd.Decoder
	chunkCache *bytes.Reader // Holds decompressed chunk data
	currentIdx int           // Current chunk index
}

func NewLobReader(reader io.ReaderAt) (*LobReader, error) {
	// Read trailer from the end of the file
	trailer, err := readTrailer(reader)
	if err != nil {
		return nil, err
	}

	lr := &LobReader{
		reader:     reader,
		trailer:    trailer,
		currentIdx: -1,
	}

	// Initialize decoder if compression is used
	if trailer.Compression == compressionMethodZstd {
		// Read dictionary
		dict := make([]byte, trailer.DictSize)
		_, err = reader.ReadAt(dict, int64(trailer.DictOffset))
		if err != nil {
			return nil, err
		}

		// Create decoder with dictionary
		decoder, err := zstd.NewReader(nil, zstd.WithDecoderDicts(dict))
		if err != nil {
			return nil, err
		}
		lr.decoder = decoder
	}

	return lr, nil
}

// GetRow returns the row at the specified index
func (lr *LobReader) GetRow(rowIndex uint64) ([]byte, error) {
	// Find which chunk contains this row
	var chunkIdx int
	var rowInChunk uint32
	var found bool

	for i, chunk := range lr.trailer.ChunkIndices {
		if uint64(rowInChunk+chunk.RowCount) > rowIndex {
			chunkIdx = i
			found = true
			break
		}
		rowInChunk += chunk.RowCount
	}

	if !found {
		return nil, errors.New("row index out of bounds")
	}

	// Load chunk if not already in cache
	if lr.currentIdx != chunkIdx {
		if err := lr.loadChunk(chunkIdx); err != nil {
			return nil, err
		}
	}

	// Calculate relative row index within chunk
	relativeIdx := uint32(rowIndex - uint64(rowInChunk))

	// Get row offset and length
	startOffset := lr.trailer.ChunkIndices[chunkIdx].RowOffsets[relativeIdx]

	// Read row length
	var rowLength uint32
	err := binary.Read(io.NewSectionReader(lr.chunkCache, int64(startOffset), 4), binary.LittleEndian, &rowLength)
	if err != nil {
		return nil, err
	}

	// Read row data
	row := make([]byte, rowLength)
	_, err = lr.chunkCache.ReadAt(row, int64(startOffset+4))
	if err != nil {
		return nil, err
	}

	return row, nil
}

func (lr *LobReader) loadChunk(index int) error {
	chunk := lr.trailer.ChunkIndices[index]

	// Read chunk header and data
	chunkData := make([]byte, chunk.Size)
	_, err := lr.reader.ReadAt(chunkData, int64(chunk.Offset))
	if err != nil {
		return err
	}

	// Parse header
	var header ChunkHeader
	headerReader := bytes.NewReader(chunkData[:24]) // header size is 24 bytes
	if err := binary.Read(headerReader, binary.LittleEndian, &header); err != nil {
		return err
	}

	// Verify CRC32
	compressedData := chunkData[24:]
	if crc32.ChecksumIEEE(compressedData) != header.CRC32 {
		return errors.New("chunk data corruption detected")
	}

	// Decompress if needed
	var uncompressedData []byte
	switch lr.trailer.Compression {
	case compressionMethodNone:
		uncompressedData = compressedData
	case compressionMethodZstd:
		uncompressedData, err = lr.decoder.DecodeAll(compressedData, nil)
		if err != nil {
			return err
		}
	default:
		return errors.New("unknown compression method")
	}

	lr.chunkCache = bytes.NewReader(uncompressedData)
	lr.currentIdx = index
	return nil
}

func readTrailer(reader io.ReaderAt) (Trailer, error) {
	// First read the last 8 bytes to verify magic number
	var magic [MagicTrailerBytesSize]byte
	if _, err := reader.ReadAt(magic[:], -MagicTrailerBytesSize); err != nil {
		return Trailer{}, err
	}
	if magic != MagicTrailer {
		return Trailer{}, errors.New("invalid file format")
	}

	// Read trailer size (8 bytes before magic)
	var trailerSize uint64
	if err := binary.Read(io.NewSectionReader(reader, -20, 8), binary.LittleEndian, &trailerSize); err != nil {
		return Trailer{}, err
	}

	// Read entire trailer
	trailerData := make([]byte, trailerSize)
	if _, err := reader.ReadAt(trailerData, -int64(trailerSize)); err != nil {
		return Trailer{}, err
	}

	var trailer Trailer
	r := bytes.NewReader(trailerData)

	// Read chunk count
	if err := binary.Read(r, binary.LittleEndian, &trailer.ChunkCount); err != nil {
		return Trailer{}, err
	}

	// Read chunk indices
	trailer.ChunkIndices = make([]ChunkIndexEntry, trailer.ChunkCount)
	for i := range trailer.ChunkIndices {
		var entry ChunkIndexEntry
		if err := binary.Read(r, binary.LittleEndian, &entry.Offset); err != nil {
			return Trailer{}, err
		}
		if err := binary.Read(r, binary.LittleEndian, &entry.Size); err != nil {
			return Trailer{}, err
		}
		if err := binary.Read(r, binary.LittleEndian, &entry.RowCount); err != nil {
			return Trailer{}, err
		}

		// Read row offsets
		var rowOffsetsCount uint32
		if err := binary.Read(r, binary.LittleEndian, &rowOffsetsCount); err != nil {
			return Trailer{}, err
		}
		entry.RowOffsets = make([]uint32, rowOffsetsCount)
		for j := range entry.RowOffsets {
			if err := binary.Read(r, binary.LittleEndian, &entry.RowOffsets[j]); err != nil {
				return Trailer{}, err
			}
		}
		trailer.ChunkIndices[i] = entry
	}

	// Read remaining trailer fields
	if err := binary.Read(r, binary.LittleEndian, &trailer.DictOffset); err != nil {
		return Trailer{}, err
	}
	if err := binary.Read(r, binary.LittleEndian, &trailer.DictSize); err != nil {
		return Trailer{}, err
	}
	if err := binary.Read(r, binary.LittleEndian, &trailer.Compression); err != nil {
		return Trailer{}, err
	}
	if err := binary.Read(r, binary.LittleEndian, &trailer.Version); err != nil {
		return Trailer{}, err
	}
	if err := binary.Read(r, binary.LittleEndian, &trailer.TrailerSize); err != nil {
		return Trailer{}, err
	}
	if err := binary.Read(r, binary.LittleEndian, &trailer.Magic); err != nil {
		return Trailer{}, err
	}

	return trailer, nil
}

func (lr *LobReader) Close() error {
	if lr.decoder != nil {
		lr.decoder.Close()
	}
	return nil
}
