package datacoord

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/msgpb"
	"github.com/milvus-io/milvus/internal/metastore/model"
	"github.com/milvus-io/milvus/internal/mocks"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/milvus-io/milvus/internal/proto/indexpb"
	"github.com/milvus-io/milvus/internal/types"
	"github.com/milvus-io/milvus/pkg/util/merr"
	"github.com/milvus-io/milvus/pkg/util/metautil"
)

func TestBroadcastAlteredCollection(t *testing.T) {
	t.Run("test server is closed", func(t *testing.T) {
		s := &Server{}
		s.stateCode.Store(commonpb.StateCode_Initializing)
		ctx := context.Background()
		resp, err := s.BroadcastAlteredCollection(ctx, nil)
		assert.NotNil(t, resp.Reason)
		assert.NoError(t, err)
	})

	t.Run("test meta non exist", func(t *testing.T) {
		s := &Server{meta: &meta{collections: make(map[UniqueID]*collectionInfo, 1)}}
		s.stateCode.Store(commonpb.StateCode_Healthy)
		ctx := context.Background()
		req := &datapb.AlterCollectionRequest{
			CollectionID: 1,
			PartitionIDs: []int64{1},
			Properties:   []*commonpb.KeyValuePair{{Key: "k", Value: "v"}},
		}
		resp, err := s.BroadcastAlteredCollection(ctx, req)
		assert.NotNil(t, resp)
		assert.NoError(t, err)
		assert.Equal(t, 1, len(s.meta.collections))
	})

	t.Run("test update meta", func(t *testing.T) {
		s := &Server{meta: &meta{collections: map[UniqueID]*collectionInfo{
			1: {ID: 1},
		}}}
		s.stateCode.Store(commonpb.StateCode_Healthy)
		ctx := context.Background()
		req := &datapb.AlterCollectionRequest{
			CollectionID: 1,
			PartitionIDs: []int64{1},
			Properties:   []*commonpb.KeyValuePair{{Key: "k", Value: "v"}},
		}

		assert.Nil(t, s.meta.collections[1].Properties)
		resp, err := s.BroadcastAlteredCollection(ctx, req)
		assert.NotNil(t, resp)
		assert.NoError(t, err)
		assert.NotNil(t, s.meta.collections[1].Properties)
	})
}

func TestServer_GcConfirm(t *testing.T) {
	t.Run("closed server", func(t *testing.T) {
		s := &Server{}
		s.stateCode.Store(commonpb.StateCode_Initializing)
		resp, err := s.GcConfirm(context.TODO(), &datapb.GcConfirmRequest{CollectionId: 100, PartitionId: 10000})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})

	t.Run("normal case", func(t *testing.T) {
		s := &Server{}
		s.stateCode.Store(commonpb.StateCode_Healthy)

		m := &meta{}
		catalog := mocks.NewDataCoordCatalog(t)
		m.catalog = catalog

		catalog.On("GcConfirm",
			mock.Anything,
			mock.AnythingOfType("int64"),
			mock.AnythingOfType("int64")).
			Return(false)

		s.meta = m

		resp, err := s.GcConfirm(context.TODO(), &datapb.GcConfirmRequest{CollectionId: 100, PartitionId: 10000})
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
		assert.False(t, resp.GetGcFinished())
	})
}

func TestGetRecoveryInfoV2(t *testing.T) {
	t.Run("test get recovery info with no segments", func(t *testing.T) {
		svr := newTestServer(t, nil)
		defer closeTestServer(t, svr)

		svr.rootCoordClientCreator = func(ctx context.Context) (types.RootCoordClient, error) {
			return newMockRootCoordClient(), nil
		}

		req := &datapb.GetRecoveryInfoRequestV2{
			CollectionID: 0,
		}
		resp, err := svr.GetRecoveryInfoV2(context.TODO(), req)
		assert.NoError(t, err)
		assert.EqualValues(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
		assert.EqualValues(t, 0, len(resp.GetSegments()))
		assert.EqualValues(t, 0, len(resp.GetChannels()))
	})

	createSegment := func(id, collectionID, partitionID, numOfRows int64, posTs uint64,
		channel string, state commonpb.SegmentState,
	) *datapb.SegmentInfo {
		return &datapb.SegmentInfo{
			ID:            id,
			CollectionID:  collectionID,
			PartitionID:   partitionID,
			InsertChannel: channel,
			NumOfRows:     numOfRows,
			State:         state,
			DmlPosition: &msgpb.MsgPosition{
				ChannelName: channel,
				MsgID:       []byte{},
				Timestamp:   posTs,
			},
			StartPosition: &msgpb.MsgPosition{
				ChannelName: "",
				MsgID:       []byte{},
				MsgGroup:    "",
				Timestamp:   0,
			},
		}
	}

	t.Run("test get earliest position of flushed segments as seek position", func(t *testing.T) {
		svr := newTestServer(t, nil)
		defer closeTestServer(t, svr)

		svr.rootCoordClientCreator = func(ctx context.Context) (types.RootCoordClient, error) {
			return newMockRootCoordClient(), nil
		}

		svr.meta.AddCollection(&collectionInfo{
			Schema: newTestSchema(),
		})

		err := svr.meta.UpdateChannelCheckpoint("vchan1", &msgpb.MsgPosition{
			ChannelName: "vchan1",
			Timestamp:   10,
			MsgID:       []byte{0, 0, 0, 0, 0, 0, 0, 0},
		})
		assert.NoError(t, err)

		err = svr.meta.CreateIndex(&model.Index{
			TenantID:     "",
			CollectionID: 0,
			FieldID:      2,
			IndexID:      0,
			IndexName:    "",
		})
		assert.NoError(t, err)

		seg1 := createSegment(0, 0, 0, 100, 10, "vchan1", commonpb.SegmentState_Flushed)
		seg1.Binlogs = []*datapb.FieldBinlog{
			{
				FieldID: 1,
				Binlogs: []*datapb.Binlog{
					{
						EntriesNum: 20,
						LogPath:    metautil.BuildInsertLogPath("a", 0, 0, 0, 1, 901),
					},
					{
						EntriesNum: 20,
						LogPath:    metautil.BuildInsertLogPath("a", 0, 0, 0, 1, 902),
					},
					{
						EntriesNum: 20,
						LogPath:    metautil.BuildInsertLogPath("a", 0, 0, 0, 1, 903),
					},
				},
			},
		}
		seg2 := createSegment(1, 0, 0, 100, 20, "vchan1", commonpb.SegmentState_Flushed)
		seg2.Binlogs = []*datapb.FieldBinlog{
			{
				FieldID: 1,
				Binlogs: []*datapb.Binlog{
					{
						EntriesNum: 30,
						LogPath:    metautil.BuildInsertLogPath("a", 0, 0, 1, 1, 801),
					},
					{
						EntriesNum: 70,
						LogPath:    metautil.BuildInsertLogPath("a", 0, 0, 1, 1, 802),
					},
				},
			},
		}
		err = svr.meta.AddSegment(context.TODO(), NewSegmentInfo(seg1))
		assert.NoError(t, err)
		err = svr.meta.AddSegment(context.TODO(), NewSegmentInfo(seg2))
		assert.NoError(t, err)
		err = svr.meta.AddSegmentIndex(&model.SegmentIndex{
			SegmentID: seg1.ID,
			BuildID:   seg1.ID,
		})
		assert.NoError(t, err)
		err = svr.meta.FinishTask(&indexpb.IndexTaskInfo{
			BuildID: seg1.ID,
			State:   commonpb.IndexState_Finished,
		})
		assert.NoError(t, err)
		err = svr.meta.AddSegmentIndex(&model.SegmentIndex{
			SegmentID: seg2.ID,
			BuildID:   seg2.ID,
		})
		assert.NoError(t, err)
		err = svr.meta.FinishTask(&indexpb.IndexTaskInfo{
			BuildID: seg2.ID,
			State:   commonpb.IndexState_Finished,
		})
		assert.NoError(t, err)

		ch := &channelMeta{Name: "vchan1", CollectionID: 0}
		svr.channelManager.AddNode(0)
		svr.channelManager.Watch(context.Background(), ch)

		req := &datapb.GetRecoveryInfoRequestV2{
			CollectionID: 0,
		}
		resp, err := svr.GetRecoveryInfoV2(context.TODO(), req)
		assert.NoError(t, err)
		assert.EqualValues(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
		assert.EqualValues(t, 1, len(resp.GetChannels()))
		assert.EqualValues(t, 0, len(resp.GetChannels()[0].GetUnflushedSegmentIds()))
		assert.ElementsMatch(t, []int64{0, 1}, resp.GetChannels()[0].GetFlushedSegmentIds())
		assert.EqualValues(t, 10, resp.GetChannels()[0].GetSeekPosition().GetTimestamp())
		assert.EqualValues(t, 2, len(resp.GetSegments()))
		// Row count corrected from 100 + 100 -> 100 + 60.
		assert.EqualValues(t, 160, resp.GetSegments()[0].GetNumOfRows()+resp.GetSegments()[1].GetNumOfRows())
	})

	t.Run("test get recovery of unflushed segments ", func(t *testing.T) {
		svr := newTestServer(t, nil)
		defer closeTestServer(t, svr)

		svr.rootCoordClientCreator = func(ctx context.Context) (types.RootCoordClient, error) {
			return newMockRootCoordClient(), nil
		}

		svr.meta.AddCollection(&collectionInfo{
			ID:     0,
			Schema: newTestSchema(),
		})

		err := svr.meta.UpdateChannelCheckpoint("vchan1", &msgpb.MsgPosition{
			ChannelName: "vchan1",
			Timestamp:   0,
			MsgID:       []byte{0, 0, 0, 0, 0, 0, 0, 0},
		})
		assert.NoError(t, err)

		seg1 := createSegment(3, 0, 0, 100, 30, "vchan1", commonpb.SegmentState_Growing)
		seg1.Binlogs = []*datapb.FieldBinlog{
			{
				FieldID: 1,
				Binlogs: []*datapb.Binlog{
					{
						EntriesNum: 20,
						LogPath:    metautil.BuildInsertLogPath("a", 0, 0, 3, 1, 901),
					},
					{
						EntriesNum: 20,
						LogPath:    metautil.BuildInsertLogPath("a", 0, 0, 3, 1, 902),
					},
					{
						EntriesNum: 20,
						LogPath:    metautil.BuildInsertLogPath("a", 0, 0, 3, 1, 903),
					},
				},
			},
		}
		seg2 := createSegment(4, 0, 0, 100, 40, "vchan1", commonpb.SegmentState_Growing)
		seg2.Binlogs = []*datapb.FieldBinlog{
			{
				FieldID: 1,
				Binlogs: []*datapb.Binlog{
					{
						EntriesNum: 30,
						LogPath:    metautil.BuildInsertLogPath("a", 0, 0, 4, 1, 801),
					},
					{
						EntriesNum: 70,
						LogPath:    metautil.BuildInsertLogPath("a", 0, 0, 4, 1, 802),
					},
				},
			},
		}
		err = svr.meta.AddSegment(context.TODO(), NewSegmentInfo(seg1))
		assert.NoError(t, err)
		err = svr.meta.AddSegment(context.TODO(), NewSegmentInfo(seg2))
		assert.NoError(t, err)

		ch := &channelMeta{Name: "vchan1", CollectionID: 0}
		svr.channelManager.AddNode(0)
		svr.channelManager.Watch(context.Background(), ch)

		req := &datapb.GetRecoveryInfoRequestV2{
			CollectionID: 0,
		}
		resp, err := svr.GetRecoveryInfoV2(context.TODO(), req)
		assert.NoError(t, err)
		assert.EqualValues(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
		assert.EqualValues(t, 0, len(resp.GetSegments()))
		assert.EqualValues(t, 1, len(resp.GetChannels()))
		assert.NotNil(t, resp.GetChannels()[0].SeekPosition)
		assert.NotEqual(t, 0, resp.GetChannels()[0].GetSeekPosition().GetTimestamp())
	})

	t.Run("test get binlogs", func(t *testing.T) {
		svr := newTestServer(t, nil)
		defer closeTestServer(t, svr)

		svr.meta.AddCollection(&collectionInfo{
			Schema: newTestSchema(),
		})

		svr.rootCoordClientCreator = func(ctx context.Context) (types.RootCoordClient, error) {
			return newMockRootCoordClient(), nil
		}

		binlogReq := &datapb.SaveBinlogPathsRequest{
			SegmentID:    0,
			CollectionID: 0,
			Field2BinlogPaths: []*datapb.FieldBinlog{
				{
					FieldID: 1,
					Binlogs: []*datapb.Binlog{
						{
							LogPath: metautil.BuildInsertLogPath("a", 0, 100, 0, 1, 801),
						},
						{
							LogPath: metautil.BuildInsertLogPath("a", 0, 100, 0, 1, 801),
						},
					},
				},
			},
			Field2StatslogPaths: []*datapb.FieldBinlog{
				{
					FieldID: 1,
					Binlogs: []*datapb.Binlog{
						{
							LogPath: metautil.BuildStatsLogPath("a", 0, 100, 0, 1000, 10000),
						},
						{
							LogPath: metautil.BuildStatsLogPath("a", 0, 100, 0, 1000, 10000),
						},
					},
				},
			},
			Deltalogs: []*datapb.FieldBinlog{
				{
					Binlogs: []*datapb.Binlog{
						{
							TimestampFrom: 0,
							TimestampTo:   1,
							LogPath:       metautil.BuildDeltaLogPath("a", 0, 100, 0, 100000),
							LogSize:       1,
						},
					},
				},
			},
		}
		segment := createSegment(0, 0, 1, 100, 10, "vchan1", commonpb.SegmentState_Flushed)
		err := svr.meta.AddSegment(context.TODO(), NewSegmentInfo(segment))
		assert.NoError(t, err)

		err = svr.meta.CreateIndex(&model.Index{
			TenantID:     "",
			CollectionID: 0,
			FieldID:      2,
			IndexID:      0,
			IndexName:    "",
		})
		assert.NoError(t, err)
		err = svr.meta.AddSegmentIndex(&model.SegmentIndex{
			SegmentID: segment.ID,
			BuildID:   segment.ID,
		})
		assert.NoError(t, err)
		err = svr.meta.FinishTask(&indexpb.IndexTaskInfo{
			BuildID: segment.ID,
			State:   commonpb.IndexState_Finished,
		})
		assert.NoError(t, err)

		err = svr.channelManager.AddNode(0)
		assert.NoError(t, err)
		err = svr.channelManager.Watch(context.Background(), &channelMeta{Name: "vchan1", CollectionID: 0})
		assert.NoError(t, err)

		sResp, err := svr.SaveBinlogPaths(context.TODO(), binlogReq)
		assert.NoError(t, err)
		assert.EqualValues(t, commonpb.ErrorCode_Success, sResp.ErrorCode)

		req := &datapb.GetRecoveryInfoRequestV2{
			CollectionID: 0,
			PartitionIDs: []int64{1},
		}
		resp, err := svr.GetRecoveryInfoV2(context.TODO(), req)
		assert.NoError(t, err)
		assert.NoError(t, merr.Error(resp.Status))
		assert.EqualValues(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
		assert.EqualValues(t, 1, len(resp.GetSegments()))
		assert.EqualValues(t, 0, resp.GetSegments()[0].GetID())
		assert.EqualValues(t, 0, len(resp.GetSegments()[0].GetBinlogs()))
	})
	t.Run("with dropped segments", func(t *testing.T) {
		svr := newTestServer(t, nil)
		defer closeTestServer(t, svr)

		svr.rootCoordClientCreator = func(ctx context.Context) (types.RootCoordClient, error) {
			return newMockRootCoordClient(), nil
		}

		svr.meta.AddCollection(&collectionInfo{
			ID:     0,
			Schema: newTestSchema(),
		})

		err := svr.meta.UpdateChannelCheckpoint("vchan1", &msgpb.MsgPosition{
			ChannelName: "vchan1",
			Timestamp:   0,
			MsgID:       []byte{0, 0, 0, 0, 0, 0, 0, 0},
		})
		assert.NoError(t, err)

		seg1 := createSegment(7, 0, 0, 100, 30, "vchan1", commonpb.SegmentState_Growing)
		seg2 := createSegment(8, 0, 0, 100, 40, "vchan1", commonpb.SegmentState_Dropped)
		err = svr.meta.AddSegment(context.TODO(), NewSegmentInfo(seg1))
		assert.NoError(t, err)
		err = svr.meta.AddSegment(context.TODO(), NewSegmentInfo(seg2))
		assert.NoError(t, err)

		ch := &channelMeta{Name: "vchan1", CollectionID: 0}
		svr.channelManager.AddNode(0)
		svr.channelManager.Watch(context.Background(), ch)

		req := &datapb.GetRecoveryInfoRequestV2{
			CollectionID: 0,
		}
		resp, err := svr.GetRecoveryInfoV2(context.TODO(), req)
		assert.NoError(t, err)
		assert.EqualValues(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
		assert.EqualValues(t, 0, len(resp.GetSegments()))
		assert.EqualValues(t, 1, len(resp.GetChannels()))
		assert.NotNil(t, resp.GetChannels()[0].SeekPosition)
		assert.NotEqual(t, 0, resp.GetChannels()[0].GetSeekPosition().GetTimestamp())
		assert.Len(t, resp.GetChannels()[0].GetDroppedSegmentIds(), 1)
		assert.Equal(t, UniqueID(8), resp.GetChannels()[0].GetDroppedSegmentIds()[0])
	})

	t.Run("with fake segments", func(t *testing.T) {
		svr := newTestServer(t, nil)
		defer closeTestServer(t, svr)

		svr.rootCoordClientCreator = func(ctx context.Context) (types.RootCoordClient, error) {
			return newMockRootCoordClient(), nil
		}

		svr.meta.AddCollection(&collectionInfo{
			ID:     0,
			Schema: newTestSchema(),
		})

		err := svr.meta.UpdateChannelCheckpoint("vchan1", &msgpb.MsgPosition{
			ChannelName: "vchan1",
			Timestamp:   0,
			MsgID:       []byte{0, 0, 0, 0, 0, 0, 0, 0},
		})
		require.NoError(t, err)

		seg1 := createSegment(7, 0, 0, 100, 30, "vchan1", commonpb.SegmentState_Growing)
		seg2 := createSegment(8, 0, 0, 100, 40, "vchan1", commonpb.SegmentState_Flushed)
		seg2.IsFake = true
		err = svr.meta.AddSegment(context.TODO(), NewSegmentInfo(seg1))
		assert.NoError(t, err)
		err = svr.meta.AddSegment(context.TODO(), NewSegmentInfo(seg2))
		assert.NoError(t, err)

		ch := &channelMeta{Name: "vchan1", CollectionID: 0}
		svr.channelManager.AddNode(0)
		svr.channelManager.Watch(context.Background(), ch)

		req := &datapb.GetRecoveryInfoRequestV2{
			CollectionID: 0,
		}
		resp, err := svr.GetRecoveryInfoV2(context.TODO(), req)
		assert.NoError(t, err)
		assert.EqualValues(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
		assert.EqualValues(t, 0, len(resp.GetSegments()))
		assert.EqualValues(t, 1, len(resp.GetChannels()))
		assert.NotNil(t, resp.GetChannels()[0].SeekPosition)
		assert.NotEqual(t, 0, resp.GetChannels()[0].GetSeekPosition().GetTimestamp())
	})

	t.Run("with failed compress", func(t *testing.T) {
		svr := newTestServer(t, nil)
		defer closeTestServer(t, svr)

		svr.rootCoordClientCreator = func(ctx context.Context) (types.RootCoordClient, error) {
			return newMockRootCoordClient(), nil
		}

		svr.meta.AddCollection(&collectionInfo{
			ID:     0,
			Schema: newTestSchema(),
		})

		err := svr.meta.UpdateChannelCheckpoint("vchan1", &msgpb.MsgPosition{
			ChannelName: "vchan1",
			Timestamp:   0,
			MsgID:       []byte{0, 0, 0, 0, 0, 0, 0, 0},
		})
		assert.NoError(t, err)

		svr.meta.AddCollection(&collectionInfo{
			ID:     1,
			Schema: newTestSchema(),
		})

		err = svr.meta.UpdateChannelCheckpoint("vchan2", &msgpb.MsgPosition{
			ChannelName: "vchan2",
			Timestamp:   0,
			MsgID:       []byte{0, 0, 0, 0, 0, 0, 0, 0},
		})
		assert.NoError(t, err)

		svr.meta.AddCollection(&collectionInfo{
			ID:     2,
			Schema: newTestSchema(),
		})

		err = svr.meta.UpdateChannelCheckpoint("vchan3", &msgpb.MsgPosition{
			ChannelName: "vchan3",
			Timestamp:   0,
			MsgID:       []byte{0, 0, 0, 0, 0, 0, 0, 0},
		})
		assert.NoError(t, err)

		svr.channelManager.AddNode(0)
		ch := &channelMeta{
			Name:         "vchan1",
			CollectionID: 0,
		}
		err = svr.channelManager.Watch(context.TODO(), ch)
		assert.NoError(t, err)

		ch = &channelMeta{
			Name:         "vchan2",
			CollectionID: 1,
		}
		err = svr.channelManager.Watch(context.TODO(), ch)
		assert.NoError(t, err)

		ch = &channelMeta{
			Name:         "vchan3",
			CollectionID: 2,
		}
		err = svr.channelManager.Watch(context.TODO(), ch)
		assert.NoError(t, err)

		seg := createSegment(8, 0, 0, 100, 40, "vchan1", commonpb.SegmentState_Flushed)
		binLogPaths := make([]*datapb.Binlog, 1)
		// miss one field
		path := metautil.JoinIDPath(0, 0, 8, fieldID)
		path = path + "/mock"
		binLogPaths[0] = &datapb.Binlog{
			EntriesNum: 10000,
			LogPath:    path,
		}

		seg.Statslogs = append(seg.Statslogs, &datapb.FieldBinlog{
			FieldID: fieldID,
			Binlogs: binLogPaths,
		})

		binLogPaths2 := make([]*datapb.Binlog, 1)
		pathCorrect := metautil.JoinIDPath(0, 0, 8, fieldID, 1)
		binLogPaths2[0] = &datapb.Binlog{
			EntriesNum: 10000,
			LogPath:    pathCorrect,
		}

		seg.Binlogs = append(seg.Binlogs, &datapb.FieldBinlog{
			FieldID: fieldID,
			Binlogs: binLogPaths2,
		})
		err = svr.meta.AddSegment(context.TODO(), NewSegmentInfo(seg))
		assert.NoError(t, err)

		// make sure collection is indexed
		err = svr.meta.CreateIndex(&model.Index{
			TenantID:        "",
			CollectionID:    0,
			FieldID:         2,
			IndexID:         0,
			IndexName:       "_default_idx_1",
			IsDeleted:       false,
			CreateTime:      0,
			TypeParams:      nil,
			IndexParams:     nil,
			IsAutoIndex:     false,
			UserIndexParams: nil,
		})
		assert.NoError(t, err)

		svr.meta.segments.SetSegmentIndex(seg.ID, &model.SegmentIndex{
			SegmentID:     seg.ID,
			CollectionID:  0,
			PartitionID:   0,
			NumRows:       100,
			IndexID:       0,
			BuildID:       0,
			NodeID:        0,
			IndexVersion:  1,
			IndexState:    commonpb.IndexState_Finished,
			FailReason:    "",
			IsDeleted:     false,
			CreateTime:    0,
			IndexFileKeys: nil,
			IndexSize:     0,
		})

		req := &datapb.GetRecoveryInfoRequestV2{
			CollectionID: 0,
		}
		_, err = svr.GetRecoveryInfoV2(context.TODO(), req)
		assert.NoError(t, err)

		// test bin log
		path = metautil.JoinIDPath(0, 0, 9, fieldID)
		path = path + "/mock"
		binLogPaths[0] = &datapb.Binlog{
			EntriesNum: 10000,
			LogPath:    path,
		}

		seg2 := createSegment(9, 1, 0, 100, 40, "vchan2", commonpb.SegmentState_Flushed)
		seg2.Binlogs = append(seg2.Binlogs, &datapb.FieldBinlog{
			FieldID: fieldID,
			Binlogs: binLogPaths,
		})
		err = svr.meta.AddSegment(context.TODO(), NewSegmentInfo(seg2))
		assert.NoError(t, err)

		// make sure collection is indexed
		err = svr.meta.CreateIndex(&model.Index{
			TenantID:        "",
			CollectionID:    1,
			FieldID:         2,
			IndexID:         1,
			IndexName:       "_default_idx_2",
			IsDeleted:       false,
			CreateTime:      0,
			TypeParams:      nil,
			IndexParams:     nil,
			IsAutoIndex:     false,
			UserIndexParams: nil,
		})
		assert.NoError(t, err)

		svr.meta.segments.SetSegmentIndex(seg2.ID, &model.SegmentIndex{
			SegmentID:     seg2.ID,
			CollectionID:  1,
			PartitionID:   0,
			NumRows:       100,
			IndexID:       1,
			BuildID:       0,
			NodeID:        0,
			IndexVersion:  1,
			IndexState:    commonpb.IndexState_Finished,
			FailReason:    "",
			IsDeleted:     false,
			CreateTime:    0,
			IndexFileKeys: nil,
			IndexSize:     0,
		})
		req = &datapb.GetRecoveryInfoRequestV2{
			CollectionID: 1,
		}
		_, err = svr.GetRecoveryInfoV2(context.TODO(), req)
		assert.NoError(t, err)
	})

	t.Run("with continuous compaction", func(t *testing.T) {
		svr := newTestServer(t, nil)
		defer closeTestServer(t, svr)

		svr.rootCoordClientCreator = func(ctx context.Context) (types.RootCoordClient, error) {
			return newMockRootCoordClient(), nil
		}

		svr.meta.AddCollection(&collectionInfo{
			ID:     0,
			Schema: newTestSchema(),
		})

		err := svr.meta.UpdateChannelCheckpoint("vchan1", &msgpb.MsgPosition{
			ChannelName: "vchan1",
			Timestamp:   0,
			MsgID:       []byte{0, 0, 0, 0, 0, 0, 0, 0},
		})
		assert.NoError(t, err)

		seg1 := createSegment(9, 0, 0, 2048, 30, "vchan1", commonpb.SegmentState_Dropped)
		seg2 := createSegment(10, 0, 0, 2048, 40, "vchan1", commonpb.SegmentState_Dropped)
		seg3 := createSegment(11, 0, 0, 2048, 40, "vchan1", commonpb.SegmentState_Dropped)
		seg3.CompactionFrom = []int64{9, 10}
		seg4 := createSegment(12, 0, 0, 2048, 40, "vchan1", commonpb.SegmentState_Dropped)
		seg5 := createSegment(13, 0, 0, 2048, 40, "vchan1", commonpb.SegmentState_Flushed)
		seg5.CompactionFrom = []int64{11, 12}
		err = svr.meta.AddSegment(context.TODO(), NewSegmentInfo(seg1))
		assert.NoError(t, err)
		err = svr.meta.AddSegment(context.TODO(), NewSegmentInfo(seg2))
		assert.NoError(t, err)
		err = svr.meta.AddSegment(context.TODO(), NewSegmentInfo(seg3))
		assert.NoError(t, err)
		err = svr.meta.AddSegment(context.TODO(), NewSegmentInfo(seg4))
		assert.NoError(t, err)
		err = svr.meta.AddSegment(context.TODO(), NewSegmentInfo(seg5))
		assert.NoError(t, err)
		err = svr.meta.CreateIndex(&model.Index{
			TenantID:        "",
			CollectionID:    0,
			FieldID:         2,
			IndexID:         0,
			IndexName:       "_default_idx_2",
			IsDeleted:       false,
			CreateTime:      0,
			TypeParams:      nil,
			IndexParams:     nil,
			IsAutoIndex:     false,
			UserIndexParams: nil,
		})
		assert.NoError(t, err)
		svr.meta.segments.SetSegmentIndex(seg4.ID, &model.SegmentIndex{
			SegmentID:     seg4.ID,
			CollectionID:  0,
			PartitionID:   0,
			NumRows:       100,
			IndexID:       0,
			BuildID:       0,
			NodeID:        0,
			IndexVersion:  1,
			IndexState:    commonpb.IndexState_Finished,
			FailReason:    "",
			IsDeleted:     false,
			CreateTime:    0,
			IndexFileKeys: nil,
			IndexSize:     0,
		})

		ch := &channelMeta{Name: "vchan1", CollectionID: 0}
		svr.channelManager.AddNode(0)
		svr.channelManager.Watch(context.Background(), ch)

		req := &datapb.GetRecoveryInfoRequestV2{
			CollectionID: 0,
		}
		resp, err := svr.GetRecoveryInfoV2(context.TODO(), req)
		assert.NoError(t, err)
		assert.EqualValues(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
		assert.NotNil(t, resp.GetChannels()[0].SeekPosition)
		assert.NotEqual(t, 0, resp.GetChannels()[0].GetSeekPosition().GetTimestamp())
		assert.Len(t, resp.GetChannels()[0].GetDroppedSegmentIds(), 0)
		assert.ElementsMatch(t, []UniqueID{}, resp.GetChannels()[0].GetUnflushedSegmentIds())
		assert.ElementsMatch(t, []UniqueID{9, 10, 12}, resp.GetChannels()[0].GetFlushedSegmentIds())
	})

	t.Run("with closed server", func(t *testing.T) {
		svr := newTestServer(t, nil)
		closeTestServer(t, svr)
		resp, err := svr.GetRecoveryInfoV2(context.TODO(), &datapb.GetRecoveryInfoRequestV2{})
		assert.NoError(t, err)
		err = merr.Error(resp.GetStatus())
		assert.ErrorIs(t, err, merr.ErrServiceNotReady)
	})
}

type GcControlServiceSuite struct {
	suite.Suite

	server *Server
}

func (s *GcControlServiceSuite) SetupTest() {
	s.server = newTestServer(s.T(), nil)
}

func (s *GcControlServiceSuite) TearDownTest() {
	if s.server != nil {
		closeTestServer(s.T(), s.server)
	}
}

func (s *GcControlServiceSuite) TestClosedServer() {
	closeTestServer(s.T(), s.server)
	resp, err := s.server.GcControl(context.TODO(), &datapb.GcControlRequest{})
	s.NoError(err)
	s.False(merr.Ok(resp))
	s.server = nil
}

func (s *GcControlServiceSuite) TestUnknownCmd() {
	resp, err := s.server.GcControl(context.TODO(), &datapb.GcControlRequest{
		Command: 0,
	})
	s.NoError(err)
	s.False(merr.Ok(resp))
}

func (s *GcControlServiceSuite) TestPause() {
	resp, err := s.server.GcControl(context.TODO(), &datapb.GcControlRequest{
		Command: datapb.GcCommand_Pause,
	})
	s.Nil(err)
	s.False(merr.Ok(resp))

	resp, err = s.server.GcControl(context.TODO(), &datapb.GcControlRequest{
		Command: datapb.GcCommand_Pause,
		Params: []*commonpb.KeyValuePair{
			{Key: "duration", Value: "not_int"},
		},
	})
	s.Nil(err)
	s.False(merr.Ok(resp))

	resp, err = s.server.GcControl(context.TODO(), &datapb.GcControlRequest{
		Command: datapb.GcCommand_Pause,
		Params: []*commonpb.KeyValuePair{
			{Key: "duration", Value: "60"},
		},
	})
	s.Nil(err)
	s.True(merr.Ok(resp))
}

func (s *GcControlServiceSuite) TestResume() {
	resp, err := s.server.GcControl(context.TODO(), &datapb.GcControlRequest{
		Command: datapb.GcCommand_Resume,
	})
	s.Nil(err)
	s.True(merr.Ok(resp))
}

func (s *GcControlServiceSuite) TestTimeoutCtx() {
	s.server.garbageCollector.close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	resp, err := s.server.GcControl(ctx, &datapb.GcControlRequest{
		Command: datapb.GcCommand_Resume,
	})
	s.Nil(err)
	s.False(merr.Ok(resp))

	resp, err = s.server.GcControl(ctx, &datapb.GcControlRequest{
		Command: datapb.GcCommand_Pause,
		Params: []*commonpb.KeyValuePair{
			{Key: "duration", Value: "60"},
		},
	})
	s.Nil(err)
	s.False(merr.Ok(resp))
}

func TestGcControlService(t *testing.T) {
	suite.Run(t, new(GcControlServiceSuite))
}
