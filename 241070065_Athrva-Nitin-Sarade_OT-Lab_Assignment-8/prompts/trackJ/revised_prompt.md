# Revised Prompt for Track J

Treat coverage preservation as a strict inequality constraint ($\sum_{t: l \in \text{Lines}(t)} x_t \ge 1 \quad \forall l$) rather than a soft objective penalty to ensure test suite minimizations never drop reachable lines.
