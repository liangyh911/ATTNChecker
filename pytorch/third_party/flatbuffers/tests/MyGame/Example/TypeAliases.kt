// automatically generated by the FlatBuffers compiler, do not modify

package MyGame.Example

import com.google.flatbuffers.BaseVector
import com.google.flatbuffers.BooleanVector
import com.google.flatbuffers.ByteVector
import com.google.flatbuffers.Constants
import com.google.flatbuffers.DoubleVector
import com.google.flatbuffers.FlatBufferBuilder
import com.google.flatbuffers.FloatVector
import com.google.flatbuffers.LongVector
import com.google.flatbuffers.StringVector
import com.google.flatbuffers.Struct
import com.google.flatbuffers.Table
import com.google.flatbuffers.UnionVector
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sign

@Suppress("unused")
@kotlin.ExperimentalUnsignedTypes
class TypeAliases : Table() {

    fun __init(_i: Int, _bb: ByteBuffer)  {
        __reset(_i, _bb)
    }
    fun __assign(_i: Int, _bb: ByteBuffer) : TypeAliases {
        __init(_i, _bb)
        return this
    }
    val i8 : Byte
        get() {
            val o = __offset(4)
            return if(o != 0) bb.get(o + bb_pos) else 0
        }
    fun mutateI8(i8: Byte) : Boolean {
        val o = __offset(4)
        return if (o != 0) {
            bb.put(o + bb_pos, i8)
            true
        } else {
            false
        }
    }
    val u8 : UByte
        get() {
            val o = __offset(6)
            return if(o != 0) bb.get(o + bb_pos).toUByte() else 0u
        }
    fun mutateU8(u8: UByte) : Boolean {
        val o = __offset(6)
        return if (o != 0) {
            bb.put(o + bb_pos, u8.toByte())
            true
        } else {
            false
        }
    }
    val i16 : Short
        get() {
            val o = __offset(8)
            return if(o != 0) bb.getShort(o + bb_pos) else 0
        }
    fun mutateI16(i16: Short) : Boolean {
        val o = __offset(8)
        return if (o != 0) {
            bb.putShort(o + bb_pos, i16)
            true
        } else {
            false
        }
    }
    val u16 : UShort
        get() {
            val o = __offset(10)
            return if(o != 0) bb.getShort(o + bb_pos).toUShort() else 0u
        }
    fun mutateU16(u16: UShort) : Boolean {
        val o = __offset(10)
        return if (o != 0) {
            bb.putShort(o + bb_pos, u16.toShort())
            true
        } else {
            false
        }
    }
    val i32 : Int
        get() {
            val o = __offset(12)
            return if(o != 0) bb.getInt(o + bb_pos) else 0
        }
    fun mutateI32(i32: Int) : Boolean {
        val o = __offset(12)
        return if (o != 0) {
            bb.putInt(o + bb_pos, i32)
            true
        } else {
            false
        }
    }
    val u32 : UInt
        get() {
            val o = __offset(14)
            return if(o != 0) bb.getInt(o + bb_pos).toUInt() else 0u
        }
    fun mutateU32(u32: UInt) : Boolean {
        val o = __offset(14)
        return if (o != 0) {
            bb.putInt(o + bb_pos, u32.toInt())
            true
        } else {
            false
        }
    }
    val i64 : Long
        get() {
            val o = __offset(16)
            return if(o != 0) bb.getLong(o + bb_pos) else 0L
        }
    fun mutateI64(i64: Long) : Boolean {
        val o = __offset(16)
        return if (o != 0) {
            bb.putLong(o + bb_pos, i64)
            true
        } else {
            false
        }
    }
    val u64 : ULong
        get() {
            val o = __offset(18)
            return if(o != 0) bb.getLong(o + bb_pos).toULong() else 0UL
        }
    fun mutateU64(u64: ULong) : Boolean {
        val o = __offset(18)
        return if (o != 0) {
            bb.putLong(o + bb_pos, u64.toLong())
            true
        } else {
            false
        }
    }
    val f32 : Float
        get() {
            val o = __offset(20)
            return if(o != 0) bb.getFloat(o + bb_pos) else 0.0f
        }
    fun mutateF32(f32: Float) : Boolean {
        val o = __offset(20)
        return if (o != 0) {
            bb.putFloat(o + bb_pos, f32)
            true
        } else {
            false
        }
    }
    val f64 : Double
        get() {
            val o = __offset(22)
            return if(o != 0) bb.getDouble(o + bb_pos) else 0.0
        }
    fun mutateF64(f64: Double) : Boolean {
        val o = __offset(22)
        return if (o != 0) {
            bb.putDouble(o + bb_pos, f64)
            true
        } else {
            false
        }
    }
    fun v8(j: Int) : Byte {
        val o = __offset(24)
        return if (o != 0) {
            bb.get(__vector(o) + j * 1)
        } else {
            0
        }
    }
    val v8Length : Int
        get() {
            val o = __offset(24); return if (o != 0) __vector_len(o) else 0
        }
    val v8AsByteBuffer : ByteBuffer get() = __vector_as_bytebuffer(24, 1)
    fun v8InByteBuffer(_bb: ByteBuffer) : ByteBuffer = __vector_in_bytebuffer(_bb, 24, 1)
    fun mutateV8(j: Int, v8: Byte) : Boolean {
        val o = __offset(24)
        return if (o != 0) {
            bb.put(__vector(o) + j * 1, v8)
            true
        } else {
            false
        }
    }
    fun vf64(j: Int) : Double {
        val o = __offset(26)
        return if (o != 0) {
            bb.getDouble(__vector(o) + j * 8)
        } else {
            0.0
        }
    }
    val vf64Length : Int
        get() {
            val o = __offset(26); return if (o != 0) __vector_len(o) else 0
        }
    val vf64AsByteBuffer : ByteBuffer get() = __vector_as_bytebuffer(26, 8)
    fun vf64InByteBuffer(_bb: ByteBuffer) : ByteBuffer = __vector_in_bytebuffer(_bb, 26, 8)
    fun mutateVf64(j: Int, vf64: Double) : Boolean {
        val o = __offset(26)
        return if (o != 0) {
            bb.putDouble(__vector(o) + j * 8, vf64)
            true
        } else {
            false
        }
    }
    companion object {
        fun validateVersion() = Constants.FLATBUFFERS_23_3_3()
        fun getRootAsTypeAliases(_bb: ByteBuffer): TypeAliases = getRootAsTypeAliases(_bb, TypeAliases())
        fun getRootAsTypeAliases(_bb: ByteBuffer, obj: TypeAliases): TypeAliases {
            _bb.order(ByteOrder.LITTLE_ENDIAN)
            return (obj.__assign(_bb.getInt(_bb.position()) + _bb.position(), _bb))
        }
        fun createTypeAliases(builder: FlatBufferBuilder, i8: Byte, u8: UByte, i16: Short, u16: UShort, i32: Int, u32: UInt, i64: Long, u64: ULong, f32: Float, f64: Double, v8Offset: Int, vf64Offset: Int) : Int {
            builder.startTable(12)
            addF64(builder, f64)
            addU64(builder, u64)
            addI64(builder, i64)
            addVf64(builder, vf64Offset)
            addV8(builder, v8Offset)
            addF32(builder, f32)
            addU32(builder, u32)
            addI32(builder, i32)
            addU16(builder, u16)
            addI16(builder, i16)
            addU8(builder, u8)
            addI8(builder, i8)
            return endTypeAliases(builder)
        }
        fun startTypeAliases(builder: FlatBufferBuilder) = builder.startTable(12)
        fun addI8(builder: FlatBufferBuilder, i8: Byte) = builder.addByte(0, i8, 0)
        fun addU8(builder: FlatBufferBuilder, u8: UByte) = builder.addByte(1, u8.toByte(), 0)
        fun addI16(builder: FlatBufferBuilder, i16: Short) = builder.addShort(2, i16, 0)
        fun addU16(builder: FlatBufferBuilder, u16: UShort) = builder.addShort(3, u16.toShort(), 0)
        fun addI32(builder: FlatBufferBuilder, i32: Int) = builder.addInt(4, i32, 0)
        fun addU32(builder: FlatBufferBuilder, u32: UInt) = builder.addInt(5, u32.toInt(), 0)
        fun addI64(builder: FlatBufferBuilder, i64: Long) = builder.addLong(6, i64, 0L)
        fun addU64(builder: FlatBufferBuilder, u64: ULong) = builder.addLong(7, u64.toLong(), 0)
        fun addF32(builder: FlatBufferBuilder, f32: Float) = builder.addFloat(8, f32, 0.0)
        fun addF64(builder: FlatBufferBuilder, f64: Double) = builder.addDouble(9, f64, 0.0)
        fun addV8(builder: FlatBufferBuilder, v8: Int) = builder.addOffset(10, v8, 0)
        fun createV8Vector(builder: FlatBufferBuilder, data: ByteArray) : Int {
            builder.startVector(1, data.size, 1)
            for (i in data.size - 1 downTo 0) {
                builder.addByte(data[i])
            }
            return builder.endVector()
        }
        fun startV8Vector(builder: FlatBufferBuilder, numElems: Int) = builder.startVector(1, numElems, 1)
        fun addVf64(builder: FlatBufferBuilder, vf64: Int) = builder.addOffset(11, vf64, 0)
        fun createVf64Vector(builder: FlatBufferBuilder, data: DoubleArray) : Int {
            builder.startVector(8, data.size, 8)
            for (i in data.size - 1 downTo 0) {
                builder.addDouble(data[i])
            }
            return builder.endVector()
        }
        fun startVf64Vector(builder: FlatBufferBuilder, numElems: Int) = builder.startVector(8, numElems, 8)
        fun endTypeAliases(builder: FlatBufferBuilder) : Int {
            val o = builder.endTable()
            return o
        }
    }
}
