# gdb -batch -x fence_log.py BINARY CORE
#
# For a core of a process run under fence.so: the fenced block the faulting
# address falls in, and every fenced block freed within a second of the
# newest free, each with what allocated and freed it and when (seconds of
# CLOCK_MONOTONIC).
import gdb

gdb.execute("set pagination off")
fault = int(gdb.parse_and_eval("$_siginfo._sifields._sigfault.si_addr"))
log = gdb.parse_and_eval("fence_log")
size = int(log.type.range()[1]) + 1


def where(address):
    address = int(address)
    if not address:
        return "-"
    named = gdb.execute(f"info symbol {address:#x}", to_string=True).strip()
    if named.startswith("No symbol"):
        named = gdb.solib_name(address) or "?"
    return f"{address:#x} {named}"


entries = [log[i] for i in range(size) if int(log[i]["base"])]
freed = [float(e["free_at"]) for e in entries if int(e["freed"])]
newest = max(freed) if freed else 0.0
print(f"fault at {fault:#x}, pc {int(gdb.selected_frame().pc()):#x}")
for e in entries:
    base, length = int(e["base"]), int(e["len"])
    hit = base <= fault < base + length
    recent = int(e["freed"]) and newest - float(e["free_at"]) < 1
    if not (hit or recent):
        continue
    print(("HIT " if hit else "    ")
          + f"block {int(e['user']):#x}, {int(e['size'])} bytes,"
          + f" allocated at {float(e['alloc_at']):.3f}, freed at {float(e['free_at']):.3f}")
    print(f"      allocated by {where(e['alloc_caller'])}")
    print(f"      freed by     {where(e['free_caller'])}")
