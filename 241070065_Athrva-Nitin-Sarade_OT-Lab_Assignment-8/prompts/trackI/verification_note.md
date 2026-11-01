# Verification Note for Track I

- Bernstein-Vazirani verified: recovered secret strings '101', '110', '011' in a single query with probability $> 0.99$.
- Deutsch-Jozsa verified: constant oracle yielded $|000\rangle$ probability $1.0$, balanced oracle yielded $0.0$.
- Grover Search verified: target state probability amplified from $0.125$ to $0.9453$ in 2 iterations ($O(\sqrt{N})$ scaling).
- Shor's algorithm order finding verified: period $r=4$ found, producing non-trivial factors $3$ and $5$ for $N=15$.
- Pytest test suite: 3/3 tests passed in 12.63s.
