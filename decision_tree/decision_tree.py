import numpy as np

# ---------------------------------------------------------------
# Test data: the 10-customer churn problem you solved by hand.
# Columns: months, tickets, uses_app.  Labels: 1 = cancelled.
# Your fit_stump should find: feature 0 (months), threshold 9,
# predict 0 if yes, 1 if no, with 9 correct.
# ---------------------------------------------------------------
X_churn = np.array([
    [2, 3, 0], [14, 0, 1], [1, 1, 0], [24, 2, 1], [6, 4, 1],
    [18, 1, 0], [3, 0, 1], [9, 3, 0], [30, 0, 0], [5, 2, 0],
])
y_churn = np.array([1, 0, 1, 0, 1, 0, 0, 1, 0, 1])

data = np.loadtxt("loans.csv", delimiter=",", skiprows=1) # creates a 2d array skipping the header row and separated by ","


X = data[:, :-1]   # every row, every column except the last
y = data[:, -1]    # every row, only the last column

def mode(y):
    """Return the most common label in y (0 or 1)."""
    # returns two arrays 
    # values = sorted labels (values = [0,1])
    # counts = how many times each one appears
    values, counts = np.unique(y, return_counts=True)
    return values[np.argmax(counts)]



def fit_stump(X, y):
    """Search and score: try every feature j and every threshold t.
    Return (j, t, y_yes, y_no, num_correct) for the best rule.
    If no split beats the baseline, return j = None."""
    #* Note: j = features, t = threshold
    best_j, best_t = None, None
    best_yes = best_no = mode(y)
    best_correct = np.sum(y == best_yes)

    # X.shape[1] = num cols
    # X.shape[0] = num rows
    for j in range(X.shape[1]):
        for t in np.unique(X[:,j]): #X[:,j] = every row in the j column sorted and excluding duplicates
            # We test each row as the threshold (the split)
            mask = X[:, j] > t # Compares every value in column j to t, returning an array of true/false values
            y_yes = y[mask] # creates an array of the true indexes in the y array
            y_no = y[~mask] # creates an array of the false indexes in the y array

            if len(y_yes) == 0 or len(y_no) == 0:
                continue
                
            yes_mode = mode(y_yes)
            no_mode = mode(y_no)

            yes_count = np.sum(y_yes == yes_mode)
            no_count = np.sum(y_no == no_mode)

            total = yes_count + no_count

            if total > best_correct:
                best_correct = total
                best_j = j
                best_t = t
                best_yes = yes_mode
                best_no = no_mode
    
    return best_j, best_t, best_yes, best_no, best_correct



    


def predict_stump(stump, X):
    """Return an array of predictions, one per row of X."""
    # TODO
    pass


# ---- Part 2 (later): the full tree ----

def fit_tree(X, y, depth, max_depth):
    """Greedy recursive splitting. Stop when the group is pure,
    no split helps, or depth == max_depth."""
    # TODO
    pass


def predict_tree(tree, x):
    """Follow the tree's if-else rules for one example x."""
    # TODO
    pass


if __name__ == "__main__":
    print(mode(np.array([1, 0, 1, 1])))      # expect 1
    print(mode(np.array([2, 0, 2, 1, 2])))   # expect 2
    print(mode(np.array([0, 1])))            # tie: what do you expect?
