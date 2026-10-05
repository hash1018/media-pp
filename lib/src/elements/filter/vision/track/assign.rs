//! Pairing what is followed with what was seen, each with at most one, at
//! the least total cost: the assignment problem, by the Hungarian method.

/// Pairs rows with columns of `cost`, each at most once, so that as many
/// pairs as can be made cost no more than `limit` and, among those, they
/// cost least in total. Pairs costing more than `limit` are not made.
///
/// `cost` is `rows` by `columns`, row-major.
pub(super) fn assign(cost: &[f64], rows: usize, columns: usize, limit: f64) -> Vec<(usize, usize)> {
    if rows == 0 || columns == 0 {
        return Vec::new();
    }
    debug_assert_eq!(cost.len(), rows * columns);
    // A pair over the limit costs more than every pair under it together,
    // so the fewest such pairs are made before cost is weighed at all.
    let refused = limit + 1.0 + rows.max(columns) as f64 * (limit.abs() + 1.0);
    let at = |row: usize, column: usize| {
        let value = cost[row * columns + column];
        if value <= limit && value.is_finite() {
            value
        } else {
            refused
        }
    };
    // The method below wants no more rows than columns; the other way round
    // is the same problem transposed.
    let transposed = rows > columns;
    let (n, m) = if transposed {
        (columns, rows)
    } else {
        (rows, columns)
    };
    let a = |i: usize, j: usize| if transposed { at(j, i) } else { at(i, j) };

    // Potentials `u` (rows) and `v` (columns), `matched[j]` the row column
    // `j` is paired with, 1-based with 0 for none — the classic O(n²m)
    // formulation with a shortest augmenting path per row.
    let mut u = vec![0.0; n + 1];
    let mut v = vec![0.0; m + 1];
    let mut matched = vec![0usize; m + 1];
    let mut way = vec![0usize; m + 1];
    for row in 1..=n {
        matched[0] = row;
        let mut column = 0;
        let mut least = vec![f64::INFINITY; m + 1];
        let mut used = vec![false; m + 1];
        loop {
            used[column] = true;
            let i = matched[column];
            let mut delta = f64::INFINITY;
            let mut next = 0;
            for j in 1..=m {
                if used[j] {
                    continue;
                }
                let reduced = a(i - 1, j - 1) - u[i] - v[j];
                if reduced < least[j] {
                    least[j] = reduced;
                    way[j] = column;
                }
                if least[j] < delta {
                    delta = least[j];
                    next = j;
                }
            }
            for j in 0..=m {
                if used[j] {
                    u[matched[j]] += delta;
                    v[j] -= delta;
                } else {
                    least[j] -= delta;
                }
            }
            column = next;
            if matched[column] == 0 {
                break;
            }
        }
        loop {
            let previous = way[column];
            matched[column] = matched[previous];
            column = previous;
            if column == 0 {
                break;
            }
        }
    }

    (1..=m)
        .filter(|&j| matched[j] != 0)
        .map(|j| {
            let (i, j) = (matched[j] - 1, j - 1);
            if transposed { (j, i) } else { (i, j) }
        })
        .filter(|&(row, column)| at(row, column) <= limit)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sorted(mut pairs: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
        pairs.sort();
        pairs
    }

    #[test]
    fn the_cheapest_pairing_is_found_not_the_greediest() {
        // Greedy takes (0, 0) at 1 and is left with (1, 1) at 10; the best
        // is (0, 1) and (1, 0) at 2 + 2.
        let cost = [1.0, 2.0, 2.0, 10.0];
        assert_eq!(sorted(assign(&cost, 2, 2, 100.0)), vec![(0, 1), (1, 0)]);
    }

    #[test]
    fn pairs_over_the_limit_are_not_made() {
        let cost = [0.1, 0.9, 0.9, 0.95];
        assert_eq!(assign(&cost, 2, 2, 0.5), vec![(0, 0)]);
    }

    #[test]
    fn more_rows_than_columns_and_more_columns_than_rows_both_pair() {
        // Three rows, two columns: the row with no cheap column is left.
        let tall = [0.2, 0.9, 0.9, 0.1, 0.3, 0.9];
        assert_eq!(sorted(assign(&tall, 3, 2, 0.5)), vec![(0, 0), (1, 1)]);
        let wide = [0.9, 0.2, 0.9, 0.1, 0.9, 0.9];
        assert_eq!(sorted(assign(&wide, 2, 3, 0.5)), vec![(0, 1), (1, 0)]);
        assert!(assign(&[], 0, 3, 0.5).is_empty());
    }

    /// The pair under the limit is kept even where pairing everything would
    /// cost less in total with refused pairs.
    #[test]
    fn as_many_pairs_as_can_be_made_under_the_limit_are_made() {
        // (0,0)=0.4 and (1,1)=0.4 both pass; (0,1)=0.0 and (1,0)=0.9 do not
        // both pass. Two pairs beat one.
        let cost = [0.4, 0.0, 0.9, 0.4];
        assert_eq!(sorted(assign(&cost, 2, 2, 0.5)), vec![(0, 0), (1, 1)]);
    }
}
