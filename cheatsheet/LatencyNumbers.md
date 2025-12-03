# Latency Numbers Everyone Should Know (2025)

A quick reference for system design interviews and back-of-the-envelope calculations.

> **Note**: These numbers are approximations for mental math. Real performance varies by hardware, workload, and conditions. Primary source: [sirupsen/napkin-math](https://github.com/sirupsen/napkin-math) benchmarks on Intel Xeon E-2236 (2024).

## CPU & Memory

| Operation | Latency |
|-----------|---------|
| L1 cache reference | 0.5 ns |
| Branch mispredict | 5 ns |
| L2 cache reference | 7 ns |
| Mutex lock/unlock | 25 ns |
| Main memory reference | 100 ns |
| Sequential memory read (64 bytes) | 0.5 ns |
| Random memory read (64 bytes) | 50 ns |

## Hashing & Crypto

| Operation | Latency |
|-----------|---------|
| Non-crypto hash (64 bytes) | 25 ns |
| Crypto-safe hash (64 bytes) | 100 ns |
| Bcrypt a password | 300 ms |

## Operating System

| Operation | Latency |
|-----------|---------|
| System call | 500 ns |
| Context switch | 10 µs |

## Compression

| Operation | Latency |
|-----------|---------|
| Compress 1KB with Snappy/Zippy | 3 µs |
| Compression throughput (general) | ~500 MiB/s |
| Decompression throughput | ~1 GiB/s |

## Storage I/O

| Operation | Latency |
|-----------|---------|
| Sequential SSD read (8KB) | 1 µs |
| Random SSD read (8KB) | 100 µs |
| Sequential SSD write (8KB, no fsync) | 10 µs |
| Sequential SSD write (8KB, +fsync) | 1 ms |
| Read 1MB sequentially from memory | 100-250 µs |
| Read 1MB sequentially from SSD | 1 ms |
| Read 1MB sequentially from HDD | 20 ms |
| HDD disk seek | 10 ms |
| Random HDD read (8KB) | 10 ms |

## Network

| Operation | Latency |
|-----------|---------|
| TCP echo server (32KB) | 10 µs |
| Proxy (Nginx/HAProxy/Envoy) | 50 µs |
| Network same zone (inside VPC) | 100 µs |
| Network within same region | 250 µs |
| Round trip within same datacenter | 500 µs |
| {Redis, Memcached, MySQL} query | 500 µs |
| Network NA Central ↔ East | 25 ms |
| Network NA Central ↔ West | 40 ms |
| Network NA East ↔ West | 60 ms |
| Network EU West ↔ NA East | 80 ms |
| Network NA West ↔ Singapore | 180 ms |
| Packet CA → Netherlands → CA | 150 ms |

## Blob Storage (S3/GCS)

| Operation | Latency |
|-----------|---------|
| GET (single connection) | 50 ms |
| PUT (single connection) | 50 ms |
| PUT (multipart, n connections) | 150 ms |

## Quick Mental Math

```
1 ns  = 10^-9 seconds
1 µs  = 10^-6 seconds = 1,000 ns
1 ms  = 10^-3 seconds = 1,000 µs = 1,000,000 ns
1 s   = 1,000 ms
```

### Orders of Magnitude

| Level | Time | Example |
|-------|------|---------|
| Nanoseconds (ns) | 1-100 ns | CPU cache, memory reference, hashing |
| Microseconds (µs) | 1-1000 µs | SSD I/O, compression, context switch, same-zone network |
| Milliseconds (ms) | 1-100 ms | Disk seeks, cross-region network, DB queries |
| Hundreds of ms | 100-1000 ms | Cross-continent RTT, bcrypt |

## Key Takeaways

1. **Memory is fast** — L1 cache is ~200x faster than main memory
2. **SSDs changed everything** — Random reads went from 10ms (HDD) to 100µs (SSD)
3. **Network is the bottleneck** — Even same-datacenter RTT is ~500µs
4. **Crypto is expensive** — Bcrypt is intentionally slow (~300ms) for security
5. **Context switches add up** — At ~10µs each, they matter at high throughput
6. **fsync is costly** — SSD writes jump from 10µs to 1ms with durability guarantees

## References

- [sirupsen/napkin-math](https://github.com/sirupsen/napkin-math) — Modern benchmarks on real hardware (primary source)
- [Napkin Math talk at SRECON](https://www.youtube.com/watch?v=IxkSlnrRFqc)
- [Systems Performance by Brendan Gregg](https://www.brendangregg.com/systems-performance-2nd-edition-book.html)
