// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package storage

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"io"
	"unsafe"

	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/pkg/v2/common"
	"github.com/milvus-io/milvus/pkg/v2/util/metautil"
)

const (
	lobLatestVersion uint32 = 1
	lobMagicNumber   uint32 = 0x4C4F4246 // "LOBF"
	lobCheckSumSize  int    = 4          // crc32
)

type LobStruct struct {
	Data   []byte
	IsLong bool
	Id     int64
}

type CompressionType uint8

const (
	CompressionNone CompressionType = iota
	CompressionZSTD
)

type lobTrailingHeader struct {
	DataLen uint64

	CollectionID int64
	PartitionID  int64
	SegmentID    int64
	FieldID      int64
	// LobID implicit in the filename

	CompressionMethodType CompressionType

	// bit 0, compressed
	// bit 1, has_checksum
	Flags       uint32
	Version     uint32
	HeaderSize  uint32
	MagicNumber uint32

	// another 4 bytes for checksum, crc32, not in the header
}

func newLobTrailingHeader(collectionID UniqueID, partitionID UniqueID, segmentID UniqueID, fieldID FieldID, dataLen uint64) *lobTrailingHeader {
	return &lobTrailingHeader{
		DataLen: dataLen,

		CollectionID: collectionID,
		PartitionID:  partitionID,
		SegmentID:    segmentID,
		FieldID:      fieldID,

		CompressionMethodType: CompressionNone,

		Flags:       0,
		Version:     lobLatestVersion,
		HeaderSize:  uint32(unsafe.Sizeof(lobTrailingHeader{})),
		MagicNumber: lobMagicNumber,
	}
}

func (h *lobTrailingHeader) serializeToBuffer(buf io.Writer) error {
	if err := binary.Write(buf, common.Endian, h); err != nil {
		return err
	}
	return nil
}

// func (h *lobTrailingHeader) serializeToBuffer(buf []byte) error {
// cursor := 0
// write := func(data interface{}) {
// 	switch v := data.(type) {
// 	case uint64:
// 		common.Endian.PutUint64(buf[cursor:], v)
// 		cursor += 8
// 	case uint32:
// 		common.Endian.PutUint32(buf[cursor:], v)
// 		cursor += 4
// 	case uint8:
// 		buf[cursor] = v
// 		cursor += 1
// 	default:
// 		panic(fmt.Sprintf("unsupported type: %T", v))
// 	}
// }

// write(h.DataLen)
// write(uint64(h.CollectionID))
// write(uint64(h.PartitionID))
// write(uint64(h.SegmentID))
// write(uint64(h.FieldID))
// write(uint8(h.CompressionMethodType))
// write(h.Flags)
// write(h.Version)
// write(h.HeaderSize)
// write(h.MagicNumber)

// return nil
// }

func serializeLob(
	collectionID UniqueID,
	partitionID UniqueID,
	segmentID UniqueID,
	fieldID FieldID,
	lobID UniqueID,
	data string,
) ([]byte, error) {
	header := newLobTrailingHeader(collectionID, partitionID, segmentID, fieldID, uint64(len(data)))
	totalSize := len(data) + int(header.HeaderSize) + lobCheckSumSize
	buf := bytes.NewBuffer(make([]byte, 0, totalSize))

	if _, err := buf.WriteString(data); err != nil {
		return nil, fmt.Errorf("failed to write data: %w", err)
	}

	if err := header.serializeToBuffer(buf); err != nil {
		return nil, fmt.Errorf("failed to write header: %w", err)
	}

	checksum := crc32.ChecksumIEEE(buf.Bytes()) // buf.Bytes() does not include the checksum
	if err := binary.Write(buf, common.Endian, checksum); err != nil {
		return nil, fmt.Errorf("failed to write checksum: %w", err)
	}

	return buf.Bytes(), nil
}

// serializeAndUploadLobs serializes and uploads LOBs to the remote storage.
// for now, it serialize LOBs one by one to minimize memory usage
// later we may optimize it by batch serialize LOBs
func SerializeAndUploadLobs(ctx context.Context, insertData []*InsertData,
	collectionID UniqueID, partitionID UniqueID, segmentID UniqueID,
	writeLobFn func(ctx context.Context, lobDir string, value []byte) error) error {
	if len(insertData) == 0 {
		return nil
	}

	for _, data := range insertData {
		for fieldID, fieldData := range data.Data {
			if fieldData.GetDataType() != schemapb.DataType_Text {
				continue
			}
			textData := fieldData.(*StringFieldData)
			if textData.IsLobs.Count() == 0 {
				// no LOB in this field
				continue
			}

			if err := ValidateLobData(textData); err != nil {
				return err
			}

			for i, lobID := range textData.LobIds {
				serializedData, err := serializeLob(collectionID, partitionID, segmentID, fieldID, lobID, textData.Data[i])
				if err != nil {
					return err
				}
				path := metautil.JoinIDPath(collectionID, partitionID, segmentID, fieldID, lobID)
				if err := writeLobFn(ctx, path, serializedData); err != nil {
					return err
				}
			}
		}
	}

	return nil
}

func ValidateLobData(textData *StringFieldData) error {
	if textData.IsLobs.Count() == 0 {
		return nil
	}
	if textData.LobIds == nil {
		return fmt.Errorf("LobIds is nil for text field")
	}
	lobCnt := 0
	for i, e := textData.IsLobs.NextSet(0); e; i, e = textData.IsLobs.NextSet(i + 1) {
		lobCnt++
		if _, exists := textData.LobIds[i]; !exists {
			return fmt.Errorf("LobId does not exist for text row %d", i)
		}
	}
	if len(textData.LobIds) != lobCnt {
		return fmt.Errorf("LobIds length mismatch with row num, %d != %d", len(textData.LobIds), lobCnt)
	}
	return nil
}
