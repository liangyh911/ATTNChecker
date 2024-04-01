// Code generated by the FlatBuffers compiler. DO NOT EDIT.

package hero

import (
	flatbuffers "github.com/google/flatbuffers/go"
)

type WarriorT struct {
	Name string `json:"name"`
	Hp uint32 `json:"hp"`
}

func (t *WarriorT) Pack(builder *flatbuffers.Builder) flatbuffers.UOffsetT {
	if t == nil { return 0 }
	nameOffset := builder.CreateString(t.Name)
	WarriorStart(builder)
	WarriorAddName(builder, nameOffset)
	WarriorAddHp(builder, t.Hp)
	return WarriorEnd(builder)
}

func (rcv *Warrior) UnPackTo(t *WarriorT) {
	t.Name = string(rcv.Name())
	t.Hp = rcv.Hp()
}

func (rcv *Warrior) UnPack() *WarriorT {
	if rcv == nil { return nil }
	t := &WarriorT{}
	rcv.UnPackTo(t)
	return t
}

type Warrior struct {
	_tab flatbuffers.Table
}

func GetRootAsWarrior(buf []byte, offset flatbuffers.UOffsetT) *Warrior {
	n := flatbuffers.GetUOffsetT(buf[offset:])
	x := &Warrior{}
	x.Init(buf, n+offset)
	return x
}

func GetSizePrefixedRootAsWarrior(buf []byte, offset flatbuffers.UOffsetT) *Warrior {
	n := flatbuffers.GetUOffsetT(buf[offset+flatbuffers.SizeUint32:])
	x := &Warrior{}
	x.Init(buf, n+offset+flatbuffers.SizeUint32)
	return x
}

func (rcv *Warrior) Init(buf []byte, i flatbuffers.UOffsetT) {
	rcv._tab.Bytes = buf
	rcv._tab.Pos = i
}

func (rcv *Warrior) Table() flatbuffers.Table {
	return rcv._tab
}

func (rcv *Warrior) Name() []byte {
	o := flatbuffers.UOffsetT(rcv._tab.Offset(4))
	if o != 0 {
		return rcv._tab.ByteVector(o + rcv._tab.Pos)
	}
	return nil
}

func (rcv *Warrior) Hp() uint32 {
	o := flatbuffers.UOffsetT(rcv._tab.Offset(6))
	if o != 0 {
		return rcv._tab.GetUint32(o + rcv._tab.Pos)
	}
	return 0
}

func (rcv *Warrior) MutateHp(n uint32) bool {
	return rcv._tab.MutateUint32Slot(6, n)
}

func WarriorStart(builder *flatbuffers.Builder) {
	builder.StartObject(2)
}
func WarriorAddName(builder *flatbuffers.Builder, name flatbuffers.UOffsetT) {
	builder.PrependUOffsetTSlot(0, flatbuffers.UOffsetT(name), 0)
}
func WarriorAddHp(builder *flatbuffers.Builder, hp uint32) {
	builder.PrependUint32Slot(1, hp, 0)
}
func WarriorEnd(builder *flatbuffers.Builder) flatbuffers.UOffsetT {
	return builder.EndObject()
}
