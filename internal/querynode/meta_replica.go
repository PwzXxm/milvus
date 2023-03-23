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

package querynode

/*
#cgo pkg-config: milvus_segcore

#include "segcore/collection_c.h"
#include "segcore/segment_c.h"
*/
import "C"
import (
	"errors"
	"fmt"
	"strconv"
	"sync"

	"github.com/milvus-io/milvus-proto/go-api/schemapb"
	"github.com/milvus-io/milvus/internal/common"
	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/metrics"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
	"github.com/milvus-io/milvus/internal/proto/querypb"
	"github.com/milvus-io/milvus/internal/util/typeutil"
	"github.com/samber/lo"
	"go.uber.org/zap"
)

var (
	ErrSegmentNotFound    = errors.New("SegmentNotFound")
	ErrPartitionNotFound  = errors.New("PartitionNotFound")
	ErrCollectionNotFound = errors.New("CollectionNotFound")
)

func WrapSegmentNotFound(segmentID int64) error {
	return fmt.Errorf("%w(%v)", ErrSegmentNotFound, segmentID)
}

func WrapCollectionNotFound(collectionID int64) error {
	return fmt.Errorf("%w(%v)", ErrCollectionNotFound, collectionID)
}

// ReplicaInterface specifies all the methods that the Collection object needs to implement in QueryNode.
// In common cases, the system has multiple query nodes. The full data of a collection will be distributed
// across multiple query nodes, and each query node's collectionReplica will maintain its own part.
type ReplicaInterface interface {
	// collection
	// getCollectionIDs returns all collection ids in the collectionReplica
	getCollectionIDs() []UniqueID
	// addCollection creates a new collection and add it to collectionReplica
	addCollection(collectionID UniqueID, schema *schemapb.CollectionSchema) *Collection
	// removeCollection removes the collection from collectionReplica
	removeCollection(collectionID UniqueID) error
	// getCollectionByID gets the collection which id is collectionID
	getCollectionByID(collectionID UniqueID) (*Collection, error)
	// hasCollection checks if collectionReplica has the collection which id is collectionID
	hasCollection(collectionID UniqueID) bool
	// getCollectionNum returns num of collections in collectionReplica
	getCollectionNum() int
	// getPartitionIDs returns partition ids of collection
	getPartitionIDs(collectionID UniqueID) ([]UniqueID, error)
	// getVecFieldIDsByCollectionID returns vector field ids of collection
	getVecFieldIDsByCollectionID(collectionID UniqueID) ([]FieldID, error)
	// getPKFieldIDsByCollectionID returns vector field ids of collection
	getPKFieldIDByCollectionID(collectionID UniqueID) (FieldID, error)
	// getSegmentInfosByColID return segments info by collectionID
	getSegmentInfosByColID(collectionID UniqueID) []*querypb.SegmentInfo
	// removeCollectionVDeltaChannel remove vdelta channel replica info from collection.
	removeCollectionVDeltaChannel(collectionID UniqueID, vDeltaChannel string)

	// partition
	// addPartition adds a new partition to collection
	addPartition(collectionID UniqueID, partitionID UniqueID) error
	// removePartition removes the partition from collectionReplica
	removePartition(partitionID UniqueID) error
	// getPartitionByID returns the partition which id is partitionID
	getPartitionByID(partitionID UniqueID) (*Partition, error)
	// hasPartition returns true if collectionReplica has the partition, false otherwise
	hasPartition(partitionID UniqueID) bool
	// getPartitionNum returns num of partitions
	getPartitionNum() int
	// getSegmentIDs returns segment ids
	getSegmentIDs(partitionID UniqueID, segType segmentType) ([]UniqueID, error)
	// getSegmentIDsByVChannel returns segment ids which virtual channel is vChannel
	// if partitionIDs is empty, it means that filtering by partitionIDs is not required.
	getSegmentIDsByVChannel(partitionIDs []UniqueID, vChannel Channel, segType segmentType) ([]UniqueID, error)

	// segment
	// addSegment add a new segment to collectionReplica
	addSegment(segmentID UniqueID, partitionID UniqueID, collectionID UniqueID, vChannelID Channel, version UniqueID, seekPosition *internalpb.MsgPosition, segType segmentType) error
	// setSegment adds a segment to collectionReplica
	setSegment(segment *Segment) error
	// removeSegment removes a segment from collectionReplica
	removeSegment(segmentID UniqueID, segType segmentType) int64
	// getSegmentByID returns the segment which id is segmentID
	getSegmentByID(segmentID UniqueID, segType segmentType) (*Segment, error)
	// hasSegment returns true if collectionReplica has the segment, false otherwise
	hasSegment(segmentID UniqueID, segType segmentType) (bool, error)
	// getSegmentNum returns num of segments in collectionReplica
	getSegmentNum(segType segmentType) int
	//  getSegmentStatistics returns the statistics of segments in collectionReplica
	getSegmentStatistics() []*internalpb.SegmentStats

	// excluded segments
	//  removeExcludedSegments will remove excludedSegments from collectionReplica
	removeExcludedSegments(collectionID UniqueID)
	// addExcludedSegments will add excludedSegments to collectionReplica
	addExcludedSegments(collectionID UniqueID, segmentInfos []*datapb.SegmentInfo)
	// getExcludedSegments returns excludedSegments of collectionReplica
	getExcludedSegments(collectionID UniqueID) ([]*datapb.SegmentInfo, error)

	// getSegmentsMemSize get the memory size in bytes of all the Segments
	getSegmentsMemSize() int64
	// freeAll will free all meta info from collectionReplica
	freeAll()
	// printReplica prints the collections, partitions and segments in the collectionReplica
	printReplica()

	// addSegmentsLoadingList add segment into black list, so get sealed segments will not return them.
	addSegmentsLoadingList(segmentIDs []UniqueID)
	// removeSegmentsLoadingList add segment into black list, so get sealed segments will not return them.
	removeSegmentsLoadingList(segmentIDs []UniqueID)

	getGrowingSegments() []*Segment
	getSealedSegments() []*Segment
	getNoSegmentChan() <-chan struct{}
}

// collectionReplica is the data replication of memory data in query node.
// It implements `ReplicaInterface` interface.
type metaReplica struct {
	mu              sync.RWMutex // guards all
	collections     map[UniqueID]*Collection
	partitions      map[UniqueID]*Partition
	growingSegments map[UniqueID]*Segment
	sealedSegments  map[UniqueID]*Segment
	noSegmentChan   chan struct{}

	excludedSegments map[UniqueID][]*datapb.SegmentInfo // map[collectionID]segmentIDs

	// segmentsBlackList stores segments which are still loading
	segmentsBlackList typeutil.UniqueSet
}

// getSegmentsMemSize get the memory size in bytes of all the Segments
func (replica *metaReplica) getSegmentsMemSize() int64 {
	replica.mu.RLock()
	defer replica.mu.RUnlock()

	memSize := int64(0)
	for _, segment := range replica.growingSegments {
		memSize += segment.getMemSize()
	}
	for _, segment := range replica.sealedSegments {
		memSize += segment.getMemSize()
	}
	return memSize
}

// printReplica prints the collections, partitions and segments in the collectionReplica
func (replica *metaReplica) printReplica() {
	replica.mu.Lock()
	defer replica.mu.Unlock()

	log.Info("collections in collectionReplica", zap.Any("info", replica.collections))
	log.Info("partitions in collectionReplica", zap.Any("info", replica.partitions))
	log.Info("growingSegments in collectionReplica", zap.Any("info", replica.growingSegments))
	log.Info("sealedSegments in collectionReplica", zap.Any("info", replica.sealedSegments))
	log.Info("excludedSegments in collectionReplica", zap.Any("info", replica.excludedSegments))
}

// ----------------------------------------------------------------------------------------------------- collection
// getCollectionIDs gets all the collection ids in the collectionReplica
func (replica *metaReplica) getCollectionIDs() []UniqueID {
	replica.mu.RLock()
	defer replica.mu.RUnlock()
	collectionIDs := make([]UniqueID, 0)
	for id := range replica.collections {
		collectionIDs = append(collectionIDs, id)
	}
	return collectionIDs
}

// addCollection creates a new collection and add it to collectionReplica
func (replica *metaReplica) addCollection(collectionID UniqueID, schema *schemapb.CollectionSchema) *Collection {
	replica.mu.Lock()
	defer replica.mu.Unlock()

	if col, ok := replica.collections[collectionID]; ok {
		return col
	}

	var newC = newCollection(collectionID, schema)
	replica.collections[collectionID] = newC
	metrics.QueryNodeNumCollections.WithLabelValues(fmt.Sprint(Params.QueryNodeCfg.GetNodeID())).Set(float64(len(replica.collections)))
	return newC
}

// removeCollection removes the collection from collectionReplica
func (replica *metaReplica) removeCollection(collectionID UniqueID) error {
	replica.mu.Lock()
	defer replica.mu.Unlock()
	return replica.removeCollectionPrivate(collectionID)
}

// removeCollectionPrivate is the private function in collectionReplica, to remove collection from collectionReplica
func (replica *metaReplica) removeCollectionPrivate(collectionID UniqueID) error {
	collection, err := replica.getCollectionByIDPrivate(collectionID)
	if err != nil {
		return err
	}

	// block incoming search&query
	collection.mu.Lock()
	defer collection.mu.Unlock()

	// delete partitions
	for _, partitionID := range collection.partitionIDs {
		// ignore error, try to delete
		_ = replica.removePartitionPrivate(partitionID)
	}

	deleteCollection(collection)
	delete(replica.collections, collectionID)

	metrics.QueryNodeNumCollections.WithLabelValues(fmt.Sprint(Params.QueryNodeCfg.GetNodeID())).Set(float64(len(replica.collections)))
	metrics.QueryNodeNumPartitions.WithLabelValues(fmt.Sprint(Params.QueryNodeCfg.GetNodeID())).Sub(float64(len(collection.partitionIDs)))
	return nil
}

// getCollectionByID gets the collection which id is collectionID
func (replica *metaReplica) getCollectionByID(collectionID UniqueID) (*Collection, error) {
	replica.mu.RLock()
	defer replica.mu.RUnlock()
	return replica.getCollectionByIDPrivate(collectionID)
}

// getCollectionByIDPrivate is the private function in collectionReplica, to get collection from collectionReplica
func (replica *metaReplica) getCollectionByIDPrivate(collectionID UniqueID) (*Collection, error) {
	collection, ok := replica.collections[collectionID]
	if !ok {
		return nil, fmt.Errorf("collection hasn't been loaded or has been released %w", WrapCollectionNotFound(collectionID))
	}

	return collection, nil
}

// hasCollection checks if collectionReplica has the collection which id is collectionID
func (replica *metaReplica) hasCollection(collectionID UniqueID) bool {
	replica.mu.RLock()
	defer replica.mu.RUnlock()
	return replica.hasCollectionPrivate(collectionID)
}

// hasCollectionPrivate is the private function in collectionReplica, to check collection in collectionReplica
func (replica *metaReplica) hasCollectionPrivate(collectionID UniqueID) bool {
	_, ok := replica.collections[collectionID]
	return ok
}

// getCollectionNum returns num of collections in collectionReplica
func (replica *metaReplica) getCollectionNum() int {
	replica.mu.RLock()
	defer replica.mu.RUnlock()
	return len(replica.collections)
}

// getPartitionIDs returns partition ids of collection
func (replica *metaReplica) getPartitionIDs(collectionID UniqueID) ([]UniqueID, error) {
	replica.mu.RLock()
	defer replica.mu.RUnlock()

	collection, err := replica.getCollectionByIDPrivate(collectionID)
	if err != nil {
		return nil, err
	}
	collection.mu.RLock()
	defer collection.mu.RUnlock()

	return collection.getPartitionIDs(), nil
}

func (replica *metaReplica) getIndexedFieldIDByCollectionIDPrivate(collectionID UniqueID, segment *Segment) ([]FieldID, error) {
	fields, err := replica.getFieldsByCollectionIDPrivate(collectionID)
	if err != nil {
		return nil, err
	}

	fieldIDS := make([]FieldID, 0)
	for _, field := range fields {
		if segment.hasLoadIndexForIndexedField(field.FieldID) {
			fieldIDS = append(fieldIDS, field.FieldID)
		}
	}
	return fieldIDS, nil
}

func (replica *metaReplica) getVecFieldIDsByCollectionIDPrivate(collectionID UniqueID) ([]FieldID, error) {
	fields, err := replica.getFieldsByCollectionIDPrivate(collectionID)
	if err != nil {
		return nil, err
	}

	vecFields := make([]FieldID, 0)
	for _, field := range fields {
		if field.DataType == schemapb.DataType_BinaryVector || field.DataType == schemapb.DataType_FloatVector {
			vecFields = append(vecFields, field.FieldID)
		}
	}
	return vecFields, nil
}

// getVecFieldIDsByCollectionID returns vector field ids of collection
func (replica *metaReplica) getVecFieldIDsByCollectionID(collectionID UniqueID) ([]FieldID, error) {
	replica.mu.RLock()
	defer replica.mu.RUnlock()

	return replica.getVecFieldIDsByCollectionIDPrivate(collectionID)
}

// getPKFieldIDsByCollectionID returns vector field ids of collection
func (replica *metaReplica) getPKFieldIDByCollectionID(collectionID UniqueID) (FieldID, error) {
	replica.mu.RLock()
	defer replica.mu.RUnlock()

	fields, err := replica.getFieldsByCollectionIDPrivate(collectionID)
	if err != nil {
		return common.InvalidFieldID, err
	}

	for _, field := range fields {
		if field.IsPrimaryKey {
			return field.FieldID, nil
		}
	}
	return common.InvalidFieldID, nil
}

// getFieldsByCollectionIDPrivate is the private function in collectionReplica, to return vector field ids of collection
func (replica *metaReplica) getFieldsByCollectionIDPrivate(collectionID UniqueID) ([]*schemapb.FieldSchema, error) {
	collection, err := replica.getCollectionByIDPrivate(collectionID)
	if err != nil {
		return nil, err
	}

	if len(collection.Schema().Fields) <= 0 {
		return nil, errors.New("no field in collection %d" + strconv.FormatInt(collectionID, 10))
	}

	return collection.Schema().Fields, nil
}

// getSegmentInfosByColID return segments info by collectionID
func (replica *metaReplica) getSegmentInfosByColID(collectionID UniqueID) []*querypb.SegmentInfo {
	replica.mu.RLock()
	defer replica.mu.RUnlock()

	segmentInfos := make([]*querypb.SegmentInfo, 0)
	_, ok := replica.collections[collectionID]
	if !ok {
		// collection not exist, so result segmentInfos is empty
		return segmentInfos
	}

	for _, segment := range replica.growingSegments {
		if segment.collectionID == collectionID {
			segmentInfo := replica.getSegmentInfo(segment)
			segmentInfos = append(segmentInfos, segmentInfo)
		}
	}
	for _, segment := range replica.sealedSegments {
		if segment.collectionID == collectionID {
			segmentInfo := replica.getSegmentInfo(segment)
			segmentInfos = append(segmentInfos, segmentInfo)
		}
	}

	return segmentInfos
}

// ----------------------------------------------------------------------------------------------------- partition
// addPartition adds a new partition to collection
func (replica *metaReplica) addPartition(collectionID UniqueID, partitionID UniqueID) error {
	replica.mu.Lock()
	defer replica.mu.Unlock()

	collection, err := replica.getCollectionByIDPrivate(collectionID)
	if err != nil {
		return err
	}
	collection.mu.Lock()
	defer collection.mu.Unlock()

	return replica.addPartitionPrivate(collection, partitionID)
}

// addPartitionPrivate is the private function in collectionReplica, to add a new partition to collection
func (replica *metaReplica) addPartitionPrivate(collection *Collection, partitionID UniqueID) error {
	if !replica.hasPartitionPrivate(partitionID) {
		collection.addPartitionID(partitionID)
		var newPartition = newPartition(collection.ID(), partitionID)
		replica.partitions[partitionID] = newPartition
		metrics.QueryNodeNumPartitions.WithLabelValues(fmt.Sprint(Params.QueryNodeCfg.GetNodeID())).Set(float64(len(replica.partitions)))
	}
	return nil
}

// removePartition removes the partition from collectionReplica
func (replica *metaReplica) removePartition(partitionID UniqueID) error {
	replica.mu.Lock()
	defer replica.mu.Unlock()

	partition, err := replica.getPartitionByIDPrivate(partitionID)
	if err != nil {
		return err
	}

	collection, err := replica.getCollectionByIDPrivate(partition.collectionID)
	if err != nil {
		return err
	}
	collection.mu.Lock()
	defer collection.mu.Unlock()

	return replica.removePartitionPrivate(partitionID)
}

// removePartitionPrivate is the private function in collectionReplica, to remove the partition from collectionReplica
// `locked` flag indicates whether corresponding collection lock is accquired before calling this method
func (replica *metaReplica) removePartitionPrivate(partitionID UniqueID) error {
	partition, err := replica.getPartitionByIDPrivate(partitionID)
	if err != nil {
		return err
	}

	collection, err := replica.getCollectionByIDPrivate(partition.collectionID)
	if err != nil {
		return err
	}

	// delete segments
	ids, _ := partition.getSegmentIDs(segmentTypeGrowing)
	for _, segmentID := range ids {
		replica.removeSegmentPrivate(segmentID, segmentTypeGrowing)
	}
	ids, _ = partition.getSegmentIDs(segmentTypeSealed)
	for _, segmentID := range ids {
		replica.removeSegmentPrivate(segmentID, segmentTypeSealed)
	}

	collection.removePartitionID(partitionID)
	delete(replica.partitions, partitionID)

	metrics.QueryNodeNumPartitions.WithLabelValues(fmt.Sprint(Params.QueryNodeCfg.GetNodeID())).Set(float64(len(replica.partitions)))
	return nil
}

// getPartitionByID returns the partition which id is partitionID
func (replica *metaReplica) getPartitionByID(partitionID UniqueID) (*Partition, error) {
	replica.mu.RLock()
	defer replica.mu.RUnlock()
	return replica.getPartitionByIDPrivate(partitionID)
}

// getPartitionByIDPrivate is the private function in collectionReplica, to get the partition
func (replica *metaReplica) getPartitionByIDPrivate(partitionID UniqueID) (*Partition, error) {
	partition, ok := replica.partitions[partitionID]
	if !ok {
		return nil, fmt.Errorf("%w(partitionID=%d)", ErrPartitionNotFound, partitionID)
	}

	return partition, nil
}

// hasPartition returns true if collectionReplica has the partition, false otherwise
func (replica *metaReplica) hasPartition(partitionID UniqueID) bool {
	replica.mu.RLock()
	defer replica.mu.RUnlock()
	return replica.hasPartitionPrivate(partitionID)
}

// hasPartitionPrivate is the private function in collectionReplica, to check if collectionReplica has the partition
func (replica *metaReplica) hasPartitionPrivate(partitionID UniqueID) bool {
	_, ok := replica.partitions[partitionID]
	return ok
}

// getPartitionNum returns num of partitions
func (replica *metaReplica) getPartitionNum() int {
	replica.mu.RLock()
	defer replica.mu.RUnlock()
	return len(replica.partitions)
}

// getSegmentIDs returns segment ids
func (replica *metaReplica) getSegmentIDs(partitionID UniqueID, segType segmentType) ([]UniqueID, error) {
	replica.mu.RLock()
	defer replica.mu.RUnlock()

	return replica.getSegmentIDsPrivate(partitionID, segType)
}

// getSegmentIDsByVChannel returns segment ids which virtual channel is vChannel
// if partitionIDs is empty, it means that filtering by partitionIDs is not required.
func (replica *metaReplica) getSegmentIDsByVChannel(partitionIDs []UniqueID, vChannel Channel, segType segmentType) ([]UniqueID, error) {
	replica.mu.RLock()
	defer replica.mu.RUnlock()

	var segments map[UniqueID]*Segment
	var ret []UniqueID

	filterPartition := len(partitionIDs) != 0
	switch segType {
	case segmentTypeGrowing:
		segments = replica.growingSegments
	case segmentTypeSealed:
		segments = replica.sealedSegments
	default:
		return nil, fmt.Errorf("unexpected segment type, segmentType = %s", segType.String())
	}

	partitionMap := make(map[UniqueID]struct{}, len(partitionIDs))
	for _, partID := range partitionIDs {
		partitionMap[partID] = struct{}{}
	}
	for _, segment := range segments {
		if segment.vChannelID == vChannel {
			if filterPartition {
				partitionID := segment.partitionID
				if _, ok := partitionMap[partitionID]; !ok {
					continue
				}
			}
			ret = append(ret, segment.ID())
		}
	}
	return ret, nil
}

// getSegmentIDsPrivate is private function in collectionReplica, it returns segment ids
func (replica *metaReplica) getSegmentIDsPrivate(partitionID UniqueID, segType segmentType) ([]UniqueID, error) {
	partition, err2 := replica.getPartitionByIDPrivate(partitionID)
	if err2 != nil {
		return nil, err2
	}

	return partition.getSegmentIDs(segType)
}

// ----------------------------------------------------------------------------------------------------- segment
// addSegment add a new segment to collectionReplica
func (replica *metaReplica) addSegment(segmentID UniqueID, partitionID UniqueID, collectionID UniqueID, vChannelID Channel, version UniqueID, seekPosition *internalpb.MsgPosition, segType segmentType) error {
	replica.mu.Lock()
	defer replica.mu.Unlock()

	collection, err := replica.getCollectionByIDPrivate(collectionID)
	if err != nil {
		return err
	}
	collection.mu.Lock()
	defer collection.mu.Unlock()

	seg, err := newSegment(collection, segmentID, partitionID, collectionID, vChannelID, segType, version, seekPosition)
	if err != nil {
		return err
	}
	return replica.addSegmentPrivate(seg)
}

// addSegmentPrivate is private function in collectionReplica, to add a new segment to collectionReplica
func (replica *metaReplica) addSegmentPrivate(segment *Segment) error {
	segID := segment.segmentID
	partition, err := replica.getPartitionByIDPrivate(segment.partitionID)
	if err != nil {
		return err
	}

	segType := segment.getType()
	ok, err := replica.hasSegmentPrivate(segID, segType)
	if err != nil {
		return err
	}
	if ok {
		return fmt.Errorf("segment has been existed, "+
			"segmentID = %d, collectionID = %d, segmentType = %s", segID, segment.collectionID, segType.String())
	}
	partition.addSegmentID(segID, segType)

	switch segType {
	case segmentTypeGrowing:
		replica.growingSegments[segID] = segment
	case segmentTypeSealed:
		replica.sealedSegments[segID] = segment
	default:
		return fmt.Errorf("unexpected segment type, segmentID = %d, segmentType = %s", segID, segType.String())
	}

	rowCount := segment.getRowCount()
	log.Info("new segment added to collection replica",
		zap.Int64("query node ID", Params.QueryNodeCfg.GetNodeID()),
		zap.Int64("collection ID", segment.collectionID),
		zap.Int64("partition ID", segment.partitionID),
		zap.Int64("segment ID", segID),
		zap.String("segment type", segType.String()),
		zap.Int64("row count", rowCount),
		zap.Uint64("segment indexed fields", segment.indexedFieldInfos.Len()),
		zap.String("vchannel", segment.vChannelID),
	)
	metrics.QueryNodeNumSegments.WithLabelValues(
		fmt.Sprint(Params.QueryNodeCfg.GetNodeID()),
		fmt.Sprint(segment.collectionID),
		fmt.Sprint(segment.partitionID),
		segType.String(),
		fmt.Sprint(segment.indexedFieldInfos.Len()),
	).Inc()
	if rowCount > 0 {
		metrics.QueryNodeNumEntities.WithLabelValues(
			fmt.Sprint(Params.QueryNodeCfg.GetNodeID()),
			fmt.Sprint(segment.collectionID),
			fmt.Sprint(segment.partitionID),
			segType.String(),
			fmt.Sprint(segment.indexedFieldInfos.Len()),
		).Add(float64(rowCount))
	}
	return nil
}

// setSegment adds a segment to collectionReplica
func (replica *metaReplica) setSegment(segment *Segment) error {
	replica.mu.Lock()
	defer replica.mu.Unlock()

	if segment == nil {
		return fmt.Errorf("nil segment when setSegment")
	}

	_, err := replica.getCollectionByIDPrivate(segment.collectionID)
	if err != nil {
		return err
	}

	return replica.addSegmentPrivate(segment)
}

// removeSegment removes a segment from collectionReplica
func (replica *metaReplica) removeSegment(segmentID UniqueID, segType segmentType) int64 {
	replica.mu.Lock()
	defer replica.mu.Unlock()

	switch segType {
	case segmentTypeGrowing:
		if segment, ok := replica.growingSegments[segmentID]; ok {
			if collection, ok := replica.collections[segment.collectionID]; ok {
				collection.mu.Lock()
				defer collection.mu.Unlock()
			}
		}
	case segmentTypeSealed:
		if segment, ok := replica.sealedSegments[segmentID]; ok {
			if collection, ok := replica.collections[segment.collectionID]; ok {
				collection.mu.Lock()
				defer collection.mu.Unlock()
			}
		}
	default:
		panic(fmt.Sprintf("unsupported segment type %s", segType.String()))
	}
	return replica.removeSegmentPrivate(segmentID, segType)
}

// removeSegmentPrivate is private function in collectionReplica, to remove a segment from collectionReplica
func (replica *metaReplica) removeSegmentPrivate(segmentID UniqueID, segType segmentType) int64 {
	var rowCount int64
	var segment *Segment
	var delta int64

	switch segType {
	case segmentTypeGrowing:
		var ok bool
		if segment, ok = replica.growingSegments[segmentID]; ok {
			if partition, ok := replica.partitions[segment.partitionID]; ok {
				partition.removeSegmentID(segmentID, segType)
			}
			rowCount = segment.getRowCount()
			delete(replica.growingSegments, segmentID)
			deleteSegment(segment)
		}
	case segmentTypeSealed:
		var ok bool
		if segment, ok = replica.sealedSegments[segmentID]; ok {
			if partition, ok := replica.partitions[segment.partitionID]; ok {
				partition.removeSegmentID(segmentID, segType)
			}
			rowCount = segment.getRowCount()
			delete(replica.sealedSegments, segmentID)
			deleteSegment(segment)
			delta++
		}
	default:
		panic(fmt.Sprintf("unsupported segment type %s", segType.String()))
	}

	if segment == nil {
		// If not found.
		log.Info("segment NOT removed from collection replica: segment not exist",
			zap.Int64("segment ID", segmentID),
			zap.String("segment type", segType.String()),
		)
	} else {
		log.Info("segment removed from collection replica",
			zap.Int64("QueryNode ID", Params.QueryNodeCfg.GetNodeID()),
			zap.Int64("collection ID", segment.collectionID),
			zap.Int64("partition ID", segment.partitionID),
			zap.Int64("segment ID", segmentID),
			zap.String("segment type", segType.String()),
			zap.Int64("row count", rowCount),
			zap.Uint64("segment indexed fields", segment.indexedFieldInfos.Len()),
		)
		metrics.QueryNodeNumSegments.WithLabelValues(
			fmt.Sprint(Params.QueryNodeCfg.GetNodeID()),
			fmt.Sprint(segment.collectionID),
			fmt.Sprint(segment.partitionID),
			segType.String(),
			// Note: this field is mutable after segment is loaded.
			fmt.Sprint(segment.indexedFieldInfos.Len()),
		).Dec()
		if rowCount > 0 {
			metrics.QueryNodeNumEntities.WithLabelValues(
				fmt.Sprint(Params.QueryNodeCfg.GetNodeID()),
				fmt.Sprint(segment.collectionID),
				fmt.Sprint(segment.partitionID),
				segType.String(),
				fmt.Sprint(segment.indexedFieldInfos.Len()),
			).Sub(float64(rowCount))
		}
	}

	replica.sendNoSegmentSignal()
	return delta
}

func (replica *metaReplica) sendNoSegmentSignal() {
	if replica.noSegmentChan == nil {
		return
	}
	select {
	case <-replica.noSegmentChan:
	default:
		if len(replica.growingSegments) == 0 && len(replica.sealedSegments) == 0 {
			close(replica.noSegmentChan)
		}
	}
}

func (replica *metaReplica) getNoSegmentChan() <-chan struct{} {
	replica.noSegmentChan = make(chan struct{})
	replica.sendNoSegmentSignal()
	return replica.noSegmentChan
}

// getSegmentByID returns the segment which id is segmentID
func (replica *metaReplica) getSegmentByID(segmentID UniqueID, segType segmentType) (*Segment, error) {
	replica.mu.RLock()
	defer replica.mu.RUnlock()
	return replica.getSegmentByIDPrivate(segmentID, segType)
}

// getSegmentByIDPrivate is private function in collectionReplica, it returns the segment which id is segmentID
func (replica *metaReplica) getSegmentByIDPrivate(segmentID UniqueID, segType segmentType) (*Segment, error) {
	switch segType {
	case segmentTypeGrowing:
		segment, ok := replica.growingSegments[segmentID]
		if !ok {
			return nil, fmt.Errorf("growing %w", WrapSegmentNotFound(segmentID))
		}
		return segment, nil
	case segmentTypeSealed:
		segment, ok := replica.sealedSegments[segmentID]
		if !ok {
			return nil, fmt.Errorf("sealed %w", WrapSegmentNotFound(segmentID))
		}
		return segment, nil
	default:
		return nil, fmt.Errorf("unexpected segment type, segmentID = %d, segmentType = %s", segmentID, segType.String())
	}
}

// hasSegment returns true if collectionReplica has the segment, false otherwise
func (replica *metaReplica) hasSegment(segmentID UniqueID, segType segmentType) (bool, error) {
	replica.mu.RLock()
	defer replica.mu.RUnlock()
	return replica.hasSegmentPrivate(segmentID, segType)
}

// hasSegmentPrivate is private function in collectionReplica, to check if collectionReplica has the segment
func (replica *metaReplica) hasSegmentPrivate(segmentID UniqueID, segType segmentType) (bool, error) {
	switch segType {
	case segmentTypeGrowing:
		_, ok := replica.growingSegments[segmentID]
		return ok, nil
	case segmentTypeSealed:
		_, ok := replica.sealedSegments[segmentID]
		return ok, nil
	default:
		return false, fmt.Errorf("unexpected segment type, segmentID = %d, segmentType = %s", segmentID, segType.String())
	}
}

// getSegmentNum returns num of segments in collectionReplica
func (replica *metaReplica) getSegmentNum(segType segmentType) int {
	replica.mu.RLock()
	defer replica.mu.RUnlock()

	switch segType {
	case segmentTypeGrowing:
		return len(replica.growingSegments)
	case segmentTypeSealed:
		return len(replica.sealedSegments)
	default:
		log.Error("unexpected segment type", zap.String("segmentType", segType.String()))
		return 0
	}
}

// getSegmentStatistics returns the statistics of segments in collectionReplica
func (replica *metaReplica) getSegmentStatistics() []*internalpb.SegmentStats {
	// TODO: deprecated
	return nil
}

// removeExcludedSegments will remove excludedSegments from collectionReplica
func (replica *metaReplica) removeExcludedSegments(collectionID UniqueID) {
	replica.mu.Lock()
	defer replica.mu.Unlock()

	delete(replica.excludedSegments, collectionID)
}

// addExcludedSegments will add excludedSegments to collectionReplica
func (replica *metaReplica) addExcludedSegments(collectionID UniqueID, segmentInfos []*datapb.SegmentInfo) {
	replica.mu.Lock()
	defer replica.mu.Unlock()

	if _, ok := replica.excludedSegments[collectionID]; !ok {
		replica.excludedSegments[collectionID] = make([]*datapb.SegmentInfo, 0)
	}

	replica.excludedSegments[collectionID] = append(replica.excludedSegments[collectionID], segmentInfos...)
}

// getExcludedSegments returns excludedSegments of collectionReplica
func (replica *metaReplica) getExcludedSegments(collectionID UniqueID) ([]*datapb.SegmentInfo, error) {
	replica.mu.RLock()
	defer replica.mu.RUnlock()

	if _, ok := replica.excludedSegments[collectionID]; !ok {
		return nil, errors.New("getExcludedSegments failed, cannot found collection, id =" + fmt.Sprintln(collectionID))
	}

	return replica.excludedSegments[collectionID], nil
}

// freeAll will free all meta info from collectionReplica
func (replica *metaReplica) freeAll() {
	replica.mu.Lock()
	defer replica.mu.Unlock()

	for id := range replica.collections {
		_ = replica.removeCollectionPrivate(id)
	}

	replica.collections = make(map[UniqueID]*Collection)
	replica.partitions = make(map[UniqueID]*Partition)
	replica.growingSegments = make(map[UniqueID]*Segment)
	replica.sealedSegments = make(map[UniqueID]*Segment)
}

func (replica *metaReplica) addSegmentsLoadingList(segmentIDs []UniqueID) {
	replica.mu.Lock()
	defer replica.mu.Unlock()

	// add to black list only segment is not loaded before
	replica.segmentsBlackList.Insert(lo.Filter(segmentIDs, func(id UniqueID, idx int) bool {
		_, isSealed := replica.sealedSegments[id]
		return !isSealed
	})...)
}

func (replica *metaReplica) removeSegmentsLoadingList(segmentIDs []UniqueID) {
	replica.mu.Lock()
	defer replica.mu.Unlock()

	replica.segmentsBlackList.Remove(segmentIDs...)
}

func (replica *metaReplica) getGrowingSegments() []*Segment {
	replica.mu.RLock()
	defer replica.mu.RUnlock()

	ret := make([]*Segment, 0, len(replica.growingSegments))
	for _, s := range replica.growingSegments {
		ret = append(ret, s)
	}
	return ret
}

func (replica *metaReplica) getSealedSegments() []*Segment {
	replica.mu.RLock()
	defer replica.mu.RUnlock()

	ret := make([]*Segment, 0, len(replica.sealedSegments))
	for _, s := range replica.sealedSegments {
		if !replica.segmentsBlackList.Contain(s.segmentID) {
			ret = append(ret, s)
		}
	}
	return ret
}

// removeCollectionVDeltaChannel remove vdelta channel replica info from collection.
func (replica *metaReplica) removeCollectionVDeltaChannel(collectionID UniqueID, vDeltaChannel string) {
	replica.mu.RLock()
	defer replica.mu.RUnlock()

	coll, ok := replica.collections[collectionID]
	if !ok {
		// if collection already release, that's fine and just return
		return
	}

	coll.removeVDeltaChannel(vDeltaChannel)
}

// newCollectionReplica returns a new ReplicaInterface
func newCollectionReplica() ReplicaInterface {
	var replica ReplicaInterface = &metaReplica{
		collections:     make(map[UniqueID]*Collection),
		partitions:      make(map[UniqueID]*Partition),
		growingSegments: make(map[UniqueID]*Segment),
		sealedSegments:  make(map[UniqueID]*Segment),

		excludedSegments: make(map[UniqueID][]*datapb.SegmentInfo),

		segmentsBlackList: make(typeutil.UniqueSet),
	}

	return replica
}

// trans segment to queryPb.segmentInfo
func (replica *metaReplica) getSegmentInfo(segment *Segment) *querypb.SegmentInfo {
	var indexName string
	var indexID int64
	var indexInfos []*querypb.FieldIndexInfo
	// TODO:: segment has multi vec column
	indexedFieldIDs, _ := replica.getIndexedFieldIDByCollectionIDPrivate(segment.collectionID, segment)
	for _, fieldID := range indexedFieldIDs {
		fieldInfo, err := segment.getIndexedFieldInfo(fieldID)
		if err == nil {
			indexName = fieldInfo.indexInfo.IndexName
			indexID = fieldInfo.indexInfo.IndexID
			indexInfos = append(indexInfos, fieldInfo.indexInfo)
		}
	}
	info := &querypb.SegmentInfo{
		SegmentID:    segment.ID(),
		CollectionID: segment.collectionID,
		PartitionID:  segment.partitionID,
		NodeID:       Params.QueryNodeCfg.GetNodeID(),
		MemSize:      segment.getMemSize(),
		NumRows:      segment.getRowCount(),
		IndexName:    indexName,
		IndexID:      indexID,
		DmChannel:    segment.vChannelID,
		SegmentState: segment.getType(),
		IndexInfos:   indexInfos,
		NodeIds:      []UniqueID{Params.QueryNodeCfg.GetNodeID()},
	}
	return info
}
