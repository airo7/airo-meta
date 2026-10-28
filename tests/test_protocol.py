from airo.protocol import cmd_packet, parse_sens


def sens(ir=(10, 20, 30), dist=1234):
    b = [0xBB, *ir, dist >> 8, dist & 0xFF, 0]
    return bytes(b + [b[1] ^ b[2] ^ b[3] ^ b[4] ^ b[5]])


def test_cmd_stop():
    assert cmd_packet(0, 0) == bytes([0xAA, 128, 128, 1, 0, 1])


def test_cmd_clips_to_pwm_max():
    assert cmd_packet(300, -300) == cmd_packet(230, -230)
    assert cmd_packet(230, -230)[1:3] == bytes([243, 13])


def test_cmd_checksum():
    p = cmd_packet(100, -40)
    assert p[5] == p[1] ^ p[2] ^ p[3]


def test_parse_roundtrip_with_garbage_and_bad_checksum():
    bad = bytearray(sens())
    bad[7] ^= 0xFF
    buf = bytearray(b"\x00\x13" + bytes(bad) + sens((1, 2, 3), 500) + sens()[:5])
    out = parse_sens(buf)
    assert [(s.ir, s.dist_mm) for s in out] == [((1, 2, 3), 500)]
    assert bytes(buf) == sens()[:5]
