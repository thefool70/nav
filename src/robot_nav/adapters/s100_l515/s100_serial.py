"""S100 差速底盘串口协议适配层（115200/8N1，无流控）。

下行命令 11 字节：0x7B 帧头、X(mm/s)、Y(=0)、Z(mrad/s) 均为 int16 大端，
byte9 为 byte0..8 逐字节 XOR，0x7D 帧尾。上行反馈 24 字节：0x7B 帧头、
0x7D 帧尾，byte22 为 byte0..21 的 XOR，byte1 停止标志、byte2..3 X(mm/s)、
byte6..7 Z(mrad/s)、byte20..21 电池电压(mV)。

本模块公开 API 统一使用 m/s、rad/s、V；只做协议收发，不包含运动控制循环。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

_FRAME_HEAD = 0x7B
_FRAME_TAIL = 0x7D
_COMMAND_LENGTH = 11
_FEEDBACK_LENGTH = 24
_INT16_MIN = -32768
_INT16_MAX = 32767
_MM_PER_M = 1000.0
_MRAD_PER_RAD = 1000.0


@dataclass(frozen=True)
class S100Status:
    """一条解析后的底盘状态反馈。单位：x_mps 为 m/s、z_radps 为 rad/s、
    battery_v 为 V；stop_flag 为 True 表示底盘停止/未使能（byte1 非 0）；
    raw_frame 为原始 24 字节反馈帧，便于排错。
    """

    stop_flag: bool
    x_mps: float
    z_radps: float
    battery_v: float
    raw_frame: bytes


@dataclass(frozen=True)
class S100SerialConfig:
    """串口连接与停止参数；协议固定使用 115200/8N1、无流控。"""

    port: str
    timeout_s: float = 0.01
    write_timeout_s: float = 0.25
    close_stop_repetitions: int = 20
    stop_interval_s: float = 0.05


def _validate_config(config: S100SerialConfig) -> None:
    """打开串口前集中校验连接与停止参数，非法时抛 ValueError。"""
    if not isinstance(config, S100SerialConfig):
        raise ValueError("config 必须为 S100SerialConfig")
    if not isinstance(config.port, str) or not config.port.strip():
        raise ValueError("port 必须为非空字符串")
    for name in ("timeout_s", "write_timeout_s", "stop_interval_s"):
        value = getattr(config, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{name} 必须为非负有限数，实际为 {value}")
    if (
        isinstance(config.close_stop_repetitions, bool)
        or not isinstance(config.close_stop_repetitions, int)
        or config.close_stop_repetitions <= 0
    ):
        raise ValueError(
            f"close_stop_repetitions 必须为正整数，实际为 {config.close_stop_repetitions}"
        )


def _checked_scaled_int16(value: float, scale: float, name: str) -> int:
    """把 value*scale 圆整为 int16 范围内的整数；bool、不可换算类型、非有限
    数或超范围时抛 ValueError。"""
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须为可换算的数值，实际为 {value}")
    try:
        converted = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} 必须为可换算的数值，实际为 {value}") from None
    if not math.isfinite(converted):
        raise ValueError(f"{name} 必须为有限数，实际为 {value}")
    scaled = round(converted * scale)
    if scaled < _INT16_MIN or scaled > _INT16_MAX:
        raise ValueError(f"{name} 换算后超出 int16 范围: {scaled}")
    return scaled


def _xor_bytes(data: bytes) -> int:
    """逐字节 XOR 求校验值。"""
    result = 0
    for byte in data:
        result ^= byte
    return result


def build_velocity_frame(x_mps: float, z_radps: float) -> bytes:
    """构造 11 字节速度命令帧。x_mps 为前进速度(m/s)，z_radps 为左转角速度
    (rad/s)，Y 固定为 0；参数非有限数或超出 int16 范围时抛出 ValueError。
    """
    x_mmps = _checked_scaled_int16(x_mps, _MM_PER_M, "x_mps")
    z_mradps = _checked_scaled_int16(z_radps, _MRAD_PER_RAD, "z_radps")
    frame = bytearray(_COMMAND_LENGTH)
    frame[0] = _FRAME_HEAD
    frame[10] = _FRAME_TAIL
    frame[3:5] = x_mmps.to_bytes(2, "big", signed=True)
    frame[5:7] = (0).to_bytes(2, "big", signed=True)
    frame[7:9] = z_mradps.to_bytes(2, "big", signed=True)
    frame[9] = _xor_bytes(bytes(frame[0:9]))
    return bytes(frame)


_STOP_FRAME = build_velocity_frame(0.0, 0.0)


def _parse_status_frame(frame: bytes) -> S100Status:
    """把 24 字节反馈帧转换为 S100Status。调用方须先校验帧头、帧尾和 XOR。"""
    x_mmps = int.from_bytes(frame[2:4], "big", signed=True)
    z_mradps = int.from_bytes(frame[6:8], "big", signed=True)
    battery_mv = int.from_bytes(frame[20:22], "big", signed=False)
    return S100Status(
        stop_flag=frame[1] != 0,
        x_mps=x_mmps / _MM_PER_M,
        z_radps=z_mradps / _MRAD_PER_RAD,
        battery_v=battery_mv / 1000.0,
        raw_frame=frame,
    )


def parse_status_frames(buffer: bytes) -> Tuple[List[S100Status], bytes]:
    """从接收缓冲解析完整的 24 字节反馈帧。

    逐字节查找帧头 0x7B；候选帧必须以 0x7D 结尾且 byte22 等于 byte0..21 的
    XOR，否则丢弃当前首字节继续寻找。返回 (已解析状态列表, 剩余未解析字节)，
    剩余部分应保留供下一次调用。
    """
    statuses: List[S100Status] = []
    offset = 0
    while offset + _FEEDBACK_LENGTH <= len(buffer):
        if buffer[offset] != _FRAME_HEAD:
            offset += 1
            continue
        frame = buffer[offset : offset + _FEEDBACK_LENGTH]
        if frame[_FEEDBACK_LENGTH - 1] != _FRAME_TAIL or _xor_bytes(
            bytes(frame[0:22])
        ) != frame[22]:
            offset += 1
            continue
        statuses.append(_parse_status_frame(frame))
        offset += _FEEDBACK_LENGTH
    return statuses, buffer[offset:]


class S100SerialConnection:
    """S100 底盘串口连接：收发命令与反馈帧并维护内部接收缓冲。

    只做协议收发，不包含运动控制循环。pyserial 在构造时才导入，方便在不
    安装 pyserial 的环境中仅使用纯函数部分。
    """

    def __init__(self, config: S100SerialConfig) -> None:
        try:
            import serial
        except ImportError as exc:
            raise ImportError("S100SerialConnection 需要 pyserial") from exc
        _validate_config(config)
        self._config = config
        self._buffer = bytearray()
        self._serial: Optional[serial.Serial] = serial.Serial(
            port=config.port,
            baudrate=115200,
            bytesize=8,
            parity="N",
            stopbits=1,
            timeout=config.timeout_s,
            write_timeout=config.write_timeout_s,
            xonxoff=False,
            rtscts=False,
            dsrdtr=False,
        )

    def read_available_statuses(self) -> List[S100Status]:
        """读取串口上已有的数据，返回解析出的全部有效状态帧（可能为空）。"""
        self._read_available()
        return self._drain_statuses()

    def wait_for_status(self, timeout_s: float) -> Optional[S100Status]:
        """等待并返回一条有效状态帧，超时返回 None。timeout_s 必须为非负有限数；
        进入后先读取一次串口，随后轮询直到截止时间；缓冲区有多条时返回最新一条。"""
        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or not math.isfinite(timeout_s)
            or timeout_s < 0
        ):
            raise ValueError(f"timeout_s 必须为非负有限数，实际为 {timeout_s}")
        deadline = time.monotonic() + timeout_s
        self._read_available()
        while True:
            statuses = self._drain_statuses()
            if statuses:
                return statuses[-1]
            if time.monotonic() >= deadline:
                return None
            self._read_available()

    def send_velocity(self, x_mps: float, z_radps: float) -> None:
        """发送速度命令：x_mps 前进速度(m/s)，z_radps 左转角速度(rad/s)。"""
        serial = self._require_open()
        serial.write(build_velocity_frame(x_mps, z_radps))
        serial.flush()

    def send_stop(self) -> None:
        """发送立即停止命令（X=0、Z=0）。"""
        serial = self._require_open()
        serial.write(_STOP_FRAME)
        serial.flush()

    def close(self) -> None:
        """按配置次数重复发送停止命令（每次 flush，间隔 stop_interval_s），
        再关闭串口；任一次写失败也仍执行最终关闭，不从此方法抛出异常。"""
        serial = self._serial
        if serial is None:
            return
        self._serial = None
        for index in range(self._config.close_stop_repetitions):
            try:
                serial.write(_STOP_FRAME)
                serial.flush()
            except Exception:
                break
            if index + 1 < self._config.close_stop_repetitions:
                time.sleep(self._config.stop_interval_s)
        try:
            serial.close()
        except Exception:
            pass

    def __enter__(self) -> S100SerialConnection:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _read_available(self) -> None:
        """把串口上已到达的数据追加到内部接收缓冲，不解析。"""
        serial = self._require_open()
        in_waiting = serial.in_waiting
        if in_waiting > 0:
            data = serial.read(in_waiting)
        else:
            data = serial.read(1)
        if data:
            self._buffer += data

    def _drain_statuses(self) -> List[S100Status]:
        """从内部缓冲解析状态帧，未解析部分保留。"""
        statuses, remaining = parse_status_frames(bytes(self._buffer))
        self._buffer = bytearray(remaining)
        return statuses

    def _require_open(self) -> Any:
        if self._serial is None:
            raise RuntimeError("串口已关闭")
        return self._serial


__all__ = [
    "S100Status",
    "S100SerialConfig",
    "S100SerialConnection",
    "build_velocity_frame",
    "parse_status_frames",
]
