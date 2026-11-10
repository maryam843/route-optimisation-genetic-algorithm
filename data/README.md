# Data

The two distance tables were supplied with the original exercise and are kept
byte for byte. Their real-world origin is unknown.

| File | Size | Nodes | Missing edges (cells equal to 100000) |
|---|---|---|---|
| `city_distances.csv` | 5,888 B | 32 | 192 |
| `city_distances_extended.csv` | 57,084 B | 100 | 1,900 |

- The first row is a header with the node numbers 0..n-1.
- Both matrices are symmetric with a zero diagonal. Real distances are
  integers from 1 to 99 (stored as floats).
- A value of 100000 means there is no road between the two nodes. The code in
  `src/route_ga/data.py` treats any value >= 100000 as a missing edge.
- Node 0 is the depot.
- The real-world origin of these numbers is unknown. They are not coordinates,
  and they are not known to come from a real city.
