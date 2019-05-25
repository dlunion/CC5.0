// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorboardX/proto/graph.proto

#ifndef PROTOBUF_tensorboardX_2fproto_2fgraph_2eproto__INCLUDED
#define PROTOBUF_tensorboardX_2fproto_2fgraph_2eproto__INCLUDED

#include <string>

#include <google/protobuf/stubs/common.h>

#if GOOGLE_PROTOBUF_VERSION < 3003000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please update
#error your headers.
#endif
#if 3003000 < GOOGLE_PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_table_driven.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/metadata.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>  // IWYU pragma: export
#include <google/protobuf/extension_set.h>  // IWYU pragma: export
#include <google/protobuf/unknown_field_set.h>
#include "tensorboardX/proto/node_def.pb.h"
#include "tensorboardX/proto/versions.pb.h"
// @@protoc_insertion_point(includes)
namespace tensorboardX {
class GraphDef;
class GraphDefDefaultTypeInternal;
extern GraphDefDefaultTypeInternal _GraphDef_default_instance_;
class NodeDef;
class NodeDefDefaultTypeInternal;
extern NodeDefDefaultTypeInternal _NodeDef_default_instance_;
class NodeDef_AttrEntry;
class NodeDef_AttrEntryDefaultTypeInternal;
extern NodeDef_AttrEntryDefaultTypeInternal _NodeDef_AttrEntry_default_instance_;
class VersionDef;
class VersionDefDefaultTypeInternal;
extern VersionDefDefaultTypeInternal _VersionDef_default_instance_;
}  // namespace tensorboardX

namespace tensorboardX {

namespace protobuf_tensorboardX_2fproto_2fgraph_2eproto {
// Internal implementation detail -- do not call these.
struct TableStruct {
  static const ::google::protobuf::internal::ParseTableField entries[];
  static const ::google::protobuf::internal::AuxillaryParseTableField aux[];
  static const ::google::protobuf::internal::ParseTable schema[];
  static const ::google::protobuf::uint32 offsets[];
  static void InitDefaultsImpl();
  static void Shutdown();
};
void AddDescriptors();
void InitDefaults();
}  // namespace protobuf_tensorboardX_2fproto_2fgraph_2eproto

// ===================================================================

class GraphDef : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:tensorboardX.GraphDef) */ {
 public:
  GraphDef();
  virtual ~GraphDef();

  GraphDef(const GraphDef& from);

  inline GraphDef& operator=(const GraphDef& from) {
    CopyFrom(from);
    return *this;
  }

  inline ::google::protobuf::Arena* GetArena() const PROTOBUF_FINAL {
    return GetArenaNoVirtual();
  }
  inline void* GetMaybeArenaPointer() const PROTOBUF_FINAL {
    return MaybeArenaPtr();
  }
  static const ::google::protobuf::Descriptor* descriptor();
  static const GraphDef& default_instance();

  static inline const GraphDef* internal_default_instance() {
    return reinterpret_cast<const GraphDef*>(
               &_GraphDef_default_instance_);
  }
  static PROTOBUF_CONSTEXPR int const kIndexInFileMessages =
    0;

  void UnsafeArenaSwap(GraphDef* other);
  void Swap(GraphDef* other);

  // implements Message ----------------------------------------------

  inline GraphDef* New() const PROTOBUF_FINAL { return New(NULL); }

  GraphDef* New(::google::protobuf::Arena* arena) const PROTOBUF_FINAL;
  void CopyFrom(const ::google::protobuf::Message& from) PROTOBUF_FINAL;
  void MergeFrom(const ::google::protobuf::Message& from) PROTOBUF_FINAL;
  void CopyFrom(const GraphDef& from);
  void MergeFrom(const GraphDef& from);
  void Clear() PROTOBUF_FINAL;
  bool IsInitialized() const PROTOBUF_FINAL;

  size_t ByteSizeLong() const PROTOBUF_FINAL;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input) PROTOBUF_FINAL;
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const PROTOBUF_FINAL;
  ::google::protobuf::uint8* InternalSerializeWithCachedSizesToArray(
      bool deterministic, ::google::protobuf::uint8* target) const PROTOBUF_FINAL;
  int GetCachedSize() const PROTOBUF_FINAL { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const PROTOBUF_FINAL;
  void InternalSwap(GraphDef* other);
  protected:
  explicit GraphDef(::google::protobuf::Arena* arena);
  private:
  static void ArenaDtor(void* object);
  inline void RegisterArenaDtor(::google::protobuf::Arena* arena);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return _internal_metadata_.arena();
  }
  inline void* MaybeArenaPtr() const {
    return _internal_metadata_.raw_arena_ptr();
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const PROTOBUF_FINAL;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // repeated .tensorboardX.NodeDef node = 1;
  int node_size() const;
  void clear_node();
  static const int kNodeFieldNumber = 1;
  const ::tensorboardX::NodeDef& node(int index) const;
  ::tensorboardX::NodeDef* mutable_node(int index);
  ::tensorboardX::NodeDef* add_node();
  ::google::protobuf::RepeatedPtrField< ::tensorboardX::NodeDef >*
      mutable_node();
  const ::google::protobuf::RepeatedPtrField< ::tensorboardX::NodeDef >&
      node() const;

  // .tensorboardX.VersionDef versions = 4;
  bool has_versions() const;
  void clear_versions();
  static const int kVersionsFieldNumber = 4;
  private:
  void _slow_mutable_versions();
  void _slow_set_allocated_versions(
      ::google::protobuf::Arena* message_arena, ::tensorboardX::VersionDef** versions);
  ::tensorboardX::VersionDef* _slow_release_versions();
  public:
  const ::tensorboardX::VersionDef& versions() const;
  ::tensorboardX::VersionDef* mutable_versions();
  ::tensorboardX::VersionDef* release_versions();
  void set_allocated_versions(::tensorboardX::VersionDef* versions);
  ::tensorboardX::VersionDef* unsafe_arena_release_versions();
  void unsafe_arena_set_allocated_versions(
      ::tensorboardX::VersionDef* versions);

  // int32 version = 3 [deprecated = true];
  GOOGLE_PROTOBUF_DEPRECATED_ATTR void clear_version();
  GOOGLE_PROTOBUF_DEPRECATED_ATTR static const int kVersionFieldNumber = 3;
  GOOGLE_PROTOBUF_DEPRECATED_ATTR ::google::protobuf::int32 version() const;
  GOOGLE_PROTOBUF_DEPRECATED_ATTR void set_version(::google::protobuf::int32 value);

  // @@protoc_insertion_point(class_scope:tensorboardX.GraphDef)
 private:

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  friend class ::google::protobuf::Arena;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::google::protobuf::RepeatedPtrField< ::tensorboardX::NodeDef > node_;
  ::tensorboardX::VersionDef* versions_;
  ::google::protobuf::int32 version_;
  mutable int _cached_size_;
  friend struct protobuf_tensorboardX_2fproto_2fgraph_2eproto::TableStruct;
};
// ===================================================================


// ===================================================================

#if !PROTOBUF_INLINE_NOT_IN_HEADERS
// GraphDef

// repeated .tensorboardX.NodeDef node = 1;
inline int GraphDef::node_size() const {
  return node_.size();
}
inline void GraphDef::clear_node() {
  node_.Clear();
}
inline const ::tensorboardX::NodeDef& GraphDef::node(int index) const {
  // @@protoc_insertion_point(field_get:tensorboardX.GraphDef.node)
  return node_.Get(index);
}
inline ::tensorboardX::NodeDef* GraphDef::mutable_node(int index) {
  // @@protoc_insertion_point(field_mutable:tensorboardX.GraphDef.node)
  return node_.Mutable(index);
}
inline ::tensorboardX::NodeDef* GraphDef::add_node() {
  // @@protoc_insertion_point(field_add:tensorboardX.GraphDef.node)
  return node_.Add();
}
inline ::google::protobuf::RepeatedPtrField< ::tensorboardX::NodeDef >*
GraphDef::mutable_node() {
  // @@protoc_insertion_point(field_mutable_list:tensorboardX.GraphDef.node)
  return &node_;
}
inline const ::google::protobuf::RepeatedPtrField< ::tensorboardX::NodeDef >&
GraphDef::node() const {
  // @@protoc_insertion_point(field_list:tensorboardX.GraphDef.node)
  return node_;
}

// .tensorboardX.VersionDef versions = 4;
inline bool GraphDef::has_versions() const {
  return this != internal_default_instance() && versions_ != NULL;
}
inline void GraphDef::clear_versions() {
  if (GetArenaNoVirtual() == NULL && versions_ != NULL) delete versions_;
  versions_ = NULL;
}
inline const ::tensorboardX::VersionDef& GraphDef::versions() const {
  // @@protoc_insertion_point(field_get:tensorboardX.GraphDef.versions)
  return versions_ != NULL ? *versions_
                         : *::tensorboardX::VersionDef::internal_default_instance();
}
inline ::tensorboardX::VersionDef* GraphDef::mutable_versions() {
  
  if (versions_ == NULL) {
    _slow_mutable_versions();
  }
  // @@protoc_insertion_point(field_mutable:tensorboardX.GraphDef.versions)
  return versions_;
}
inline ::tensorboardX::VersionDef* GraphDef::release_versions() {
  // @@protoc_insertion_point(field_release:tensorboardX.GraphDef.versions)
  
  if (GetArenaNoVirtual() != NULL) {
    return _slow_release_versions();
  } else {
    ::tensorboardX::VersionDef* temp = versions_;
    versions_ = NULL;
    return temp;
  }
}
inline  void GraphDef::set_allocated_versions(::tensorboardX::VersionDef* versions) {
  ::google::protobuf::Arena* message_arena = GetArenaNoVirtual();
  if (message_arena == NULL) {
    delete versions_;
  }
  if (versions != NULL) {
    _slow_set_allocated_versions(message_arena, &versions);
  }
  versions_ = versions;
  if (versions) {
    
  } else {
    
  }
  // @@protoc_insertion_point(field_set_allocated:tensorboardX.GraphDef.versions)
}

// int32 version = 3 [deprecated = true];
inline void GraphDef::clear_version() {
  version_ = 0;
}
inline ::google::protobuf::int32 GraphDef::version() const {
  // @@protoc_insertion_point(field_get:tensorboardX.GraphDef.version)
  return version_;
}
inline void GraphDef::set_version(::google::protobuf::int32 value) {
  
  version_ = value;
  // @@protoc_insertion_point(field_set:tensorboardX.GraphDef.version)
}

#endif  // !PROTOBUF_INLINE_NOT_IN_HEADERS

// @@protoc_insertion_point(namespace_scope)


}  // namespace tensorboardX

// @@protoc_insertion_point(global_scope)

#endif  // PROTOBUF_tensorboardX_2fproto_2fgraph_2eproto__INCLUDED