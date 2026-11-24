#!/usr/bin/env python3
"""Drive the KV260 full adder from a Jupyter kernel or the command line."""

import argparse
from pathlib import Path
import gc

from pynq import Overlay

DEFAULT_BITSTREAM = "Overlays/example.bit"
DEFAULT_AB_GPIO = "axi_gpio_0"
DEFAULT_CIN_GPIO = "axi_gpio_1"
DEFAULT_RESULT_GPIO = "axi_gpio_2"


def get_ip(overlay: Overlay, name: str):
    try:
        return getattr(overlay, name)
    except AttributeError as exc:
        available = ", ".join(sorted(overlay.ip_dict))
        raise RuntimeError(
            f"AXI IP '{name}' was not found. Available IP names: {available}"
        ) from exc


class FullAdder:
    def __init__(
        self,
        overlay: Overlay,
        ab_gpio_name: str = DEFAULT_AB_GPIO,
        cin_gpio_name: str = DEFAULT_CIN_GPIO,
        result_gpio_name: str = DEFAULT_RESULT_GPIO,
    ) -> None:
        self.overlay = overlay
        ab = get_ip(overlay, ab_gpio_name)
        cin = get_ip(overlay, cin_gpio_name)
        result = get_ip(overlay, result_gpio_name)
        self.a = ab.channel1
        self.b = ab.channel2
        self.cin = cin.channel1
        self.sum = result.channel1
        self.cout = result.channel2

    def calculate(self, a: int, b: int, cin: int) -> tuple[int, int]:
        for name, value in (("A", a), ("B", b), ("Cin", cin)):
            if value not in (0, 1):
                raise ValueError(f"{name} must be 0 or 1")

        self.a.write(a, 1)
        self.b.write(b, 1)
        self.cin.write(cin, 1)
        return int(self.sum.read()) & 1, int(self.cout.read()) & 1

    def free(self):
        self.overlay.free()


def load_adder(bitstream: str | Path) -> FullAdder:
    """Download the selected bitstream once and bind its AXI GPIO channels."""
    path = Path(bitstream).expanduser().resolve()
    if path.suffix.lower() != ".bit":
        raise ValueError(f"Select a .bit file: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Bitstream not found: {path}")
    hwh = path.with_suffix(".hwh")
    if not hwh.is_file():
        raise FileNotFoundError(f"Place the matching Vivado hardware description at {hwh}")
    return FullAdder(Overlay(str(path), download=True))


def print_test(adder: FullAdder, a: int, b: int, cin: int) -> bool:
    """Test the currently loaded adder and return whether the outputs match."""
    sum_bit, carry_out = adder.calculate(a, b, cin)
    expected_sum = a ^ b ^ cin
    expected_carry = (a & b) | (a & cin) | (b & cin)
    passed = (sum_bit, carry_out) == (expected_sum, expected_carry)
    status = "PASS" if passed else "FAIL"
    print(f"A={a}  B={b}  Cin={cin}  ->  Sum={sum_bit}  Cout={carry_out}  [{status}]")
    return passed

def free_resources(adder: FullAdder) -> bool:
    adder.free()
    del adder
    gc.collet()
