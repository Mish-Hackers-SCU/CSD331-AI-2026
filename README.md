## 🧭 CSD331 AI Practical Guide

This section provides quick access to the main topics and concepts covered in this revision guide. Click on any link to jump directly to that section.

| Chapter Title | Content Focus |
| :--- | :--- |
| [**🚀 NumPy Essentials**](#numpy-essentials) | Array creation, shaping, statistical functions, and conditional assignment. |
| [**📊 Pandas Data Manipulation**](#pandas-data-manipulation) | DataFrame creation, indexing (`loc`/`iloc`), cleaning (`fillna`), and transformation (`apply`, `get_dummies`). |
| [**🗃️ NumPy Array vs. Pandas Series**](#numpy-array-vs-pandas-series) | Comparison of 1D data structures (ndarray and Series) and their use cases. |
| **Machine Learning Concepts (Supervised & Unsupervised)** | Core algorithms and techniques for model building. |
|       **Supervised Learning** | Algorithms trained on labeled data. |
|           ➡️ [**Regression Analysis**](#regression-analysis-modeling-relationships) | Predicting continuous values (The core section on this topic). |
|               _Linear, Poly, Multi Regression_ | Detailed breakdown of regression models, $R^2$, and residuals. |
|           ➡️ [**Classification**](#classification-algorithms-predicting-categories) | Predicting discrete categorical labels. |
|               *Decision Tree, Random Forest, KNN* | Gini impurity, Majority Voting, Distance-based Classification and N-neighbors. |
|       **Unsupervised Learning** | Algorithms for pattern discovery in unlabeled data. |
|           ➡️ [**Clustering**](#unsupervised-learning-clustering-algorithms) | Grouping similar data points. |
|               *K-means, Hierarchical* | Centroids, Dendograms. |

-----


### 🚀 NumPy Essentials

NumPy is the fundamental package for scientific computing in Python. It provides a high-performance multidimensional array object, and tools for working with these arrays.

### Array Creation: `np.array()`, `np.asarray()`, `np.arange()`, `np.linspace()`, `np.zeros()`, `np.ones()`, `np.empty()`, `np.identity()`

| Method | Purpose | Differentiation/Context |
| :--- | :--- | :--- |
| `np.array()` | Creates a NumPy array from any array-like object (list, tuple). | **Always makes a copy** of the input data if the input is already a NumPy array. |
| `np.asarray()` | Converts input to an array. | **Does not make a copy** if the input is already an `ndarray` of the same `dtype`. This saves memory/time when possible. |
| `np.arange()` | Creates arrays with regularly spaced values within a given interval. | Similar to Python's `range()`, but returns an `ndarray`. Requires a **step** value. |
| `np.linspace()` | Creates arrays with a specified number of samples, evenly spaced, over a closed interval. | Requires the **number of samples** (e.g., `num=50`) instead of a step size. Excellent for visualization and testing. |
| `np.zeros()` | Creates an array of a given shape and type, filled with zeros. | |
| `np.ones()` | Creates an array of a given shape and type, filled with ones. | |
| `np.empty()` | Creates an array without initializing its entries (contains uninitialized garbage values). | **Fastest** way to create an array, but the contents are non-deterministic (whatever was in that memory location). |
| `np.identity()` | Returns the square identity array of size $n \times n$. | A special case of `np.eye()` where $k=0$ (main diagonal). |

**Code Example:**

```python
import numpy as np

# np.array vs np.asarray
list_a = [1, 2, 3]
arr_a = np.array(list_a)
arr_b = np.asarray(arr_a)

# Modify the original array 'arr_a'
arr_a[0] = 99 

print(f"Original list: {list_a}")
print(f"np.array (copy behavior): {arr_a}")
print(f"np.asarray (no-copy behavior): {arr_b}") # arr_b is a view of arr_a, hence it is modified too.
# The list_a is NOT modified as np.array makes a copy.

print("\nnp.arange vs np.linspace:")
print(f"np.arange(0, 10, 2): {np.arange(0, 10, 2)}") # Start, Stop (exclusive), Step
print(f"np.linspace(0, 10, 5): {np.linspace(0, 10, 5)}") # Start, Stop (inclusive), Number of samples

print("\nSpecial Arrays:")
print(f"np.zeros((2, 2)): \n{np.zeros((2, 2))}")
print(f"np.empty((2, 2)): \n{np.empty((2, 2))}")
print(f"np.identity(3): \n{np.identity(3)}")
````

**Output:**

```
Original list: [1, 2, 3]
np.array (copy behavior): [99  2  3]
np.asarray (no-copy behavior): [99  2  3]

np.arange vs np.linspace:
np.arange(0, 10, 2): [0 2 4 6 8]
np.linspace(0, 10, 5): [ 0.   2.5  5.   7.5 10. ]

Special Arrays:
np.zeros((2, 2)): 
[[0. 0.]
 [0. 0.]]
np.empty((2, 2)): 
[[...garbage values...]]
np.identity(3): 
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
```

### Array Information and Shaping: `np_arr.shape`, `np_arr.reshape()`

| Method | Purpose | Notes |
| :--- | :--- | :--- |
| `np_arr.shape` | Attribute that returns the dimensions of the array. | Essential for understanding array structure and debugging. |
| `np_arr.reshape()` | Gives a new shape to an array without changing its data. | The total number of elements must remain the same. Using `-1` in one dimension lets NumPy calculate the size. |

**Code Example:**

```python
arr_orig = np.arange(12)
print(f"Original shape: {arr_orig.shape}")

arr_reshaped = arr_orig.reshape(3, 4)
print(f"Reshaped 3x4 array:\n{arr_reshaped}")

arr_auto_reshaped = arr_orig.reshape(6, -1) # NumPy calculates the second dimension as 2
print(f"Auto-calculated shape:\n{arr_auto_reshaped}")
```

**Output:**

```
Original shape: (12,)
Reshaped 3x4 array:
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
Auto-calculated shape:
[[ 0  1]
 [ 2  3]
 [ 4  5]
 [ 6  7]
 [ 8  9]
 [10 11]]
```

### Statistical and Conditional Functions: `np.mean()`, `np.gradient()`, `np.select()`

| Method | Purpose | Context/Use Case |
| :--- | :--- | :--- |
| `np.mean()` | Computes the arithmetic mean along the specified axis. | Quick calculation of average values. Crucial for normalization and feature scaling. |
| `np.gradient()` | Returns the gradient of an $N$-dimensional array. | Used to estimate the derivative (rate of change) of a function represented by the array values. Useful in image processing (edge detection) or physical simulations. |
| `np.select()` | Assigns values to array elements based on a set of conditions. | Extremely powerful for conditional assignment, similar to nested `if/elif/else` statements or SQL's `CASE` in an array operation. |

**Code Example:**

```python
arr_data = np.array([[10, 20, 30], [40, 50, 60]])

print(f"Mean of all elements: {np.mean(arr_data)}")
print(f"Mean along rows (axis=1): {np.mean(arr_data, axis=1)}")

# np.gradient example: Approximating the derivative of a discrete function
func_values = np.array([0, 1, 4, 9, 16]) # The function f(x) = x^2 at x=0, 1, 2, 3, 4
grad = np.gradient(func_values)
print(f"\nFunction values: {func_values}")
print(f"Gradient (approx. derivative): {grad}") 
# The derivative of x^2 is 2x. At the endpoints, the gradient is less accurate.

# np.select example: Conditional assignment
conditions = [
    arr_data > 50,
    (arr_data >= 20) & (arr_data <= 50)
]
choices = [
    'High',
    'Medium'
]
default_val = 'Low'

result_arr = np.select(conditions, choices, default=default_val)
print(f"\nConditional assignment (np.select):\n{result_arr}")
```

**Output:**

```
Mean of all elements: 35.0
Mean along rows (axis=1): [20. 50.]

Function values: [ 0  1  4  9 16]
Gradient (approx. derivative): [ 1.   2.5  4.   5.5  7. ]

Conditional assignment (np.select):
[['Low' 'Medium' 'Medium']
 ['Medium' 'Medium' 'High']]
```

### Sample Code Breakdown: Structured Arrays and Slicing

Structured arrays allow you to have elements with different data types (like a table row).

**Code Example:**

```python
# 1. Structured Array (Simulating a simple database table)
dt = np.dtype([('name', 'S10'), ('age', 'i4'), ('salary', 'f8')])
data = [('Alice', 30, 70000.0), ('Bob', 25, 60000.0)]
structured_arr = np.array(data, dtype=dt)
print(f"Structured Array:\n{structured_arr}")
print(f"Accessing 'name' column: {structured_arr['name']}")

# 2. np.append (Avoid using in performance-critical code; better to pre-allocate)
new_row = np.array(('Charlie', 40, 90000.0), dtype=dt)
appended_arr = np.append(structured_arr, new_row)
print(f"\nAppended Array (flattened): {appended_arr}") # Note: np.append can flatten if axis is not specified

# 3. Array Slicing (e.g., arr_square[:, 0:2])
arr_square = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
slice_result = arr_square[:, 0:2] # All rows, columns 0 and 1 (2 is exclusive)
print(f"\nOriginal Square Array:\n{arr_square}")
print(f"Slice [:, 0:2]:\n{slice_result}")
```

**Output:**

```
Structured Array:
[(b'Alice', 30, 70000.) (b'Bob', 25, 60000.)]
Accessing 'name' column: [b'Alice' b'Bob']

Appended Array (flattened): [(b'Alice', 30, 70000.) (b'Bob', 25, 60000.) (b'Charlie', 40, 90000.)]

Original Square Array:
[[1 2 3]
 [4 5 6]
 [7 8 9]]
Slice [:, 0:2]:
[[1 2]
 [4 5]
 [7 8]]
```

-----

## 📊 Pandas Data Manipulation

Pandas is a fast, powerful, flexible, and easy-to-use open-source data analysis and manipulation tool, built on top of the Python programming language.

## 🗃️ NumPy Array vs. Pandas Series

<img width="1600" height="800" alt="image" src="https://github.com/user-attachments/assets/df804688-3425-4e07-841f-84880ff94100" />


While both the NumPy `ndarray` (array) and the Pandas `Series` represent a one-dimensional sequence of data, they are designed for slightly different purposes and offer distinct functionalities. Understanding their relationship is crucial, as the Pandas Series is built directly upon the foundation of the NumPy array.

| Feature | NumPy `ndarray` | Pandas `Series` |
| :--- | :--- | :--- |
| **Dimensionality** | 1D, 2D, or N-dimensional | Always 1D |
| **Indexing** | Implicit integer indexing (0, 1, 2, ...). | **Explicit, labeled indexing** (can use numbers, strings, dates, etc.). |
| **Data Types** | Must be homogeneous (all elements of the same type). | Can be heterogeneous, but typically homogeneous for performance. |
| **Mutability** | Size and values are mutable. | Size is immutable, values are mutable. |
| **Context** | Core for numerical computations, matrix operations, and backend for ML libraries. | Core for data manipulation, cleaning, and time-series analysis; acts like a labeled column in a DataFrame. |
| **Missing Values** | Supports `np.nan` (Not a Number). | Supports `np.nan` and has extensive built-in methods (`fillna`, `dropna`). |

### NumPy Array (Core Data Structure)

The NumPy array is the basic building block. It offers optimized, low-level data storage and mathematical operations, focusing purely on numerical efficiency.

**Code Example (NumPy Array):**

```python
import numpy as np
# Homogeneous data, implicit integer index
arr = np.array([10, 20, 30, 40])
print(f"NumPy Array:\n{arr}")
print(f"Access value at index 1: {arr[1]}") 
# Indexing is position-based (0, 1, 2, ...)
```

**Output:**

```
NumPy Array:
[10 20 30 40]
Access value at index 1: 20
```

### Pandas Series (Labeled Data Structure)

The Pandas Series is essentially a labeled NumPy array. It adds an **Index** object (the labels) to the array's data, enabling easier data alignment and retrieval using meaningful names rather than just integer positions.

**Code Example (Pandas Series):**

```python
import pandas as pd
# Labeled data, explicit string index
s = pd.Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'])
print(f"Pandas Series:\n{s}")
print(f"Access value at label 'b': {s['b']}") 
# Indexing is label-based
print(f"Access value at position 1 (iloc): {s.iloc[1]}")
```

**Output:**

```
Pandas Series:
a    10
b    20
c    30
d    40
dtype: int64
Access value at label 'b': 20
Access value at position 1 (iloc): 20
```

### Data Overview and Structuring: `pd.DataFrame()`, `df.info()`, `df.describe()`

| Method | Purpose | Differentiation/Context |
| :--- | :--- | :--- |
| `pd.DataFrame()` | Primary data structure, representing data in rows and columns (like a spreadsheet or SQL table). | Core object for all Pandas operations. Can be created from lists, dicts, NumPy arrays, etc. |
| `df.info()` | Prints a concise summary of a DataFrame. | Crucial for checking **data types**, **memory usage**, and **non-null counts** (identifying missing data). |
| `df.describe()` | Generates descriptive statistics of the DataFrame's **numerical columns**. | Shows count, mean, std, min, max, and quartiles. Quickly assess data distribution and range. |

**Code Example:**

```python
import pandas as pd

data = {'A': [1, 2, 3, 4], 
        'B': [1.0, 2.5, np.nan, 4.0], 
        'C': ['cat', 'dog', 'cat', 'fish']}
df = pd.DataFrame(data)

print("df.info() output:")
df.info()

print("\ndf.describe() output:")
print(df.describe())
```

**Output:**

```
df.info() output:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 4 entries, 0 to 3
Data columns (total 3 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   A       4 non-null      int64  
 1   B       3 non-null      float64
 2   C       4 non-null      object 
dtypes: float64(1), int64(1), object(1)
memory usage: 228.0+ bytes

df.describe() output:
             A         B
count  4.00000  3.000000
mean   2.50000  2.500000
std    1.29099  1.500000
min    1.00000  1.000000
25%    1.75000  1.750000
50%    2.50000  2.500000
75%    3.25000  3.250000
max    4.00000  4.000000
```

### Data Indexing and Selection: `df.loc[]`, `df.iloc[]`

| Method | Purpose | Differentiation/Context |
| :--- | :--- | :--- |
| `df.loc[]` | **Label-based** indexing. Selects data using **explicit index and column names**. | Preferred for semantic selection and when the index is meaningful (e.g., dates). |
| `df.iloc[]` | **Integer-based** indexing. Selects data using **integer position** (0 to $N-1$) for both rows and columns. | Useful when the position is more relevant, such as when slicing an array. |

**Code Example:**

```python
data_loc_iloc = {'Name': ['Sam', 'Tom', 'Ray'], 'Score': [85, 92, 78]}
df_idx = pd.DataFrame(data_loc_iloc, index=['s1', 's2', 's3'])

print("Original DataFrame:\n", df_idx)

# df.loc[]: Accessing by label (index name and column name)
print("\ndf.loc['s2', 'Score']: ", df_idx.loc['s2', 'Score'])
print("df.loc[:, ['Name']]:\n", df_idx.loc[:, ['Name']])

# df.iloc[]: Accessing by integer position (row 1, column 0)
print("\ndf.iloc[1, 0]: ", df_idx.iloc[1, 0])
print("df.iloc[0:2, :]:\n", df_idx.iloc[0:2, :]) # Rows 0 and 1, all columns
```

**Output:**

```
Original DataFrame:
    Name  Score
s1  Sam     85
s2  Tom     92
s3  Ray     78

df.loc['s2', 'Score']:  92
df.loc[:, ['Name']]:
     Name
s1   Sam
s2   Tom
s3   Ray

df.iloc[1, 0]:  Tom
df.iloc[0:2, :]:
    Name  Score
s1  Sam     85
s2  Tom     92
```

### Data Cleaning and Transformation: `df.apply()`, `df.fillna()`, `df.drop()`, `df.drop_duplicates()`

| Method | Purpose | Context/Use Case |
| :--- | :--- | :--- |
| `df.apply()` | Applies a function along an axis of the DataFrame (row or column). | Excellent for element-wise or row/column-wise transformations using `lambda` functions or custom functions. |
| `df.fillna()` | Fills missing values (`NaN`) using specified methods. | Crucial for imputation. Common methods include: **`value`**, **`method='ffill'`** (forward fill), **`method='bfill'`** (backward fill), **`df['col'].mean()`** (impute with mean). |
| `df.drop()` | Removes specified labels from rows or columns. | Used for removing unneeded features (`axis=1`) or observations (`axis=0`). |
| `df.drop_duplicates()` | Returns a DataFrame with duplicate rows removed. | Standard procedure in data cleaning to ensure unique observations. |

**Code Example:**

```python
df_clean = pd.DataFrame({
    'ID': [1, 2, 2, 3, 4],
    'Val': [10, 20, 20, 30, np.nan],
    'Cat': ['A', 'B', 'B', 'C', 'A']
})
print("Initial DataFrame:\n", df_clean)

# df.drop_duplicates()
df_nodup = df_clean.drop_duplicates()
print("\nAfter drop_duplicates():\n", df_nodup)

# df.drop() - drop a column
df_dropped = df_nodup.drop(columns=['ID'], inplace=False)
print("\nAfter dropping 'ID' column:\n", df_dropped)

# df.fillna() - ffill and mean imputation
df_filled = df_dropped.copy()
# 1. Forward Fill for 'Val' (NaN at index 4 gets the value from index 3)
df_filled['Val'] = df_filled['Val'].fillna(method='ffill') 
print("\nAfter ffill on 'Val':\n", df_filled)

# df.apply() - simple lambda function for element-wise transformation
df_applied = df_filled.copy()
df_applied['Val_Squared'] = df_applied['Val'].apply(lambda x: x**2)
print("\nAfter df.apply() (Val_Squared):\n", df_applied)
```

**Output:**

```
Initial DataFrame:
    ID   Val Cat
0   1  10.0   A
1   2  20.0   B
2   2  20.0   B
3   3  30.0   C
4   4   NaN   A

After drop_duplicates():
    ID   Val Cat
0   1  10.0   A
1   2  20.0   B
3   3  30.0   C
4   4   NaN   A

After dropping 'ID' column:
    Val Cat
0  10.0   A
1  20.0   B
3  30.0   C
4   NaN   A

After ffill on 'Val':
    Val Cat
0  10.0   A
1  20.0   B
3  30.0   C
4  30.0   A

After df.apply() (Val_Squared):
    Val Cat  Val_Squared
0  10.0   A        100.0
1  20.0   B        400.0
3  30.0   C        900.0
4  30.0   A        900.0
```

### Data Aggregation and Encoding: `pd.concat()`, `pd.get_dummies()`, `pd.pivot_table()`

| Method | Purpose | Context/Use Case |
| :--- | :--- | :--- |
| `pd.concat()` | Concatenates pandas objects along a particular axis. | Primary way to stack DataFrames (rows-wise: `axis=0`) or join columns (column-wise: `axis=1`). |
| `pd.get_dummies()` | Converts categorical data into dummy variables (One-Hot Encoding). | **Essential** pre-processing step for ML algorithms that cannot handle categorical features directly. |
| `pd.pivot_table()` | Creates a spreadsheet-style pivot table as a DataFrame. | Powerful for aggregation, summarizing data by groups. Requires specifying `index`, `columns`, `values`, and an `aggfunc`. |

**Code Example:**

```python
df1 = pd.DataFrame({'X': [1, 2], 'Y': ['A', 'B']})
df2 = pd.DataFrame({'X': [3, 4], 'Y': ['C', 'D']})
df_cat = pd.DataFrame({'Feature': ['Red', 'Green', 'Red', 'Blue'], 'Value': [1, 2, 3, 4]})

# pd.concat() - row-wise
df_combined = pd.concat([df1, df2], ignore_index=True)
print("Row-wise pd.concat():\n", df_combined)

# pd.get_dummies() - One-Hot Encoding
df_encoded = pd.get_dummies(df_cat, columns=['Feature'], prefix='Color')
print("\nOne-Hot Encoding (pd.get_dummies()):\n", df_encoded)

# pd.pivot_table() - simple aggregation
data_pivot = {'City': ['NY', 'NY', 'LA', 'LA'], 'Category': ['A', 'B', 'A', 'B'], 'Sales': [100, 150, 200, 250]}
df_pivot = pd.DataFrame(data_pivot)
pivot_result = pd.pivot_table(df_pivot, values='Sales', index='City', columns='Category', aggfunc=np.sum)
print("\npd.pivot_table (Sum of Sales by City and Category):\n", pivot_result)
```

**Output:**

```
Row-wise pd.concat():
    X  Y
0  1  A
1  2  B
2  3  C
3  4  D

One-Hot Encoding (pd.get_dummies()):
   Value  Color_Blue  Color_Green  Color_Red
0      1       False        False       True
1      2       False         True      False
2      3       False        False       True
3      4        True        False      False

pd.pivot_table (Sum of Sales by City and Category):
 Category    A    B
City             
LA        200  250
NY        100  150
```

### Sample Code Integration: NumPy and Pandas Synergy

This example shows the power of using NumPy's conditional assignment (`np.select()`) directly on a Pandas Series for creating a new categorical feature.

**Code Example:**

```python
# Create a sample DataFrame
df_synergy = pd.DataFrame({
    'ID': [101, 102, 103, 104, 105],
    'Score': [65, 92, 45, 78, 88]
})

# Define conditions based on the Pandas Series 'Score'
conditions = [
    df_synergy['Score'] >= 90,
    df_synergy['Score'] >= 70,
    df_synergy['Score'] >= 50
]

# Define corresponding choices
choices = [
    'Excellent',
    'Good',
    'Pass'
]

# Use np.select on the Series, assigning the result to a new DataFrame column
df_synergy['Grade'] = np.select(conditions, choices, default='Fail')

print("DataFrame with new 'Grade' column created via np.select:\n", df_synergy)
```

**Output:**

```
DataFrame with new 'Grade' column created via np.select:
    ID  Score      Grade
0  101     65       Pass
1  102     92  Excellent
2  103     45       Fail
3  104     78       Good
4  105     88       Good
```

-----
## 📈 Regression Analysis: Modeling Relationships

Regression analysis is a fundamental statistical process for estimating the relationships among variables. It helps in understanding how the typical value of the dependent variable (target) changes when any one of the independent variables (features) is varied.

### 1\. Simple Linear Regression

Simple Linear Regression (SLR) is used to model the relationship between two continuous variables: one independent variable ($X$) and one dependent variable ($Y$). It assumes the relationship can be approximated by a straight line, defined by the equation:

$$Y = \beta_0 + \beta_1 X + \epsilon$$

Where:

  * $\mathbf{Y}$: The dependent variable (Target).
  * $\mathbf{X}$: The independent variable (Feature).
  * $\mathbf{\beta_0}$ (Intercept): The value of $Y$ when $X=0$.
  * $\mathbf{\beta_1}$ (Slope): The change in $Y$ for a one-unit change in $X$.
  * $\mathbf{\epsilon}$: The error term (residual).

![Fitting_gif](https://docs.eaqbe.com/assets/images/linear_regression_animation-de67583e136349fe9096eaa31ea14871.gif)

#### **Fitting the Line (Using `scipy.stats.linregress`)**

The `scipy.stats.linregress` function is excellent for simple linear regression as it calculates the slope, intercept, R-value, p-value, and standard error quickly.

**Code Example:**

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 1. Sample Data (X: hours studied, Y: exam score)
X = np.array([2, 4, 6, 8, 10, 12])
Y = np.array([55, 65, 75, 80, 85, 95])

# 2. Fit the line using linregress
slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)

# 3. Create the regression line for plotting
Y_pred = intercept + slope * X

print(f"Slope (B1): {slope:.2f}")
print(f"Intercept (B0): {intercept:.2f}")

# Plotting
plt.figure(figsize=(8, 5))
plt.scatter(X, Y, color='blue', label='Actual Data')
plt.plot(X, Y_pred, color='red', label=f'Regression Line\n(R-squared: {r_value**2:.3f})')
plt.title('Simple Linear Regression')
plt.xlabel('Hours Studied (X)')
plt.ylabel('Exam Score (Y)')
plt.legend()
plt.grid(True)
plt.show()
```

**Code Explanation & Output Interpretation:**

  * `stats.linregress(X, Y)`: Performs the least-squares fitting.
  * **Slope (B1): 3.86**: For every additional hour studied, the score increases by approximately 3.86 points.
  * **Intercept (B0): 48.81**: The predicted score for 0 hours studied.
  * **R-squared (0.975)**: Indicates a very strong fit, as the regression line explains 97.5% of the variance in the exam scores.

### 2\. Polynomial Regression

Polynomial Regression models the relationship between the independent variable $X$ and the dependent variable $Y$ as an $n^{th}$-degree polynomial. This is useful when the relationship is non-linear but can be "straightened out" by using powers of $X$.

$$\hat{Y} = \beta_0 + \beta_1 X + \beta_2 X^2 + \dots + \beta_n X^n + \epsilon$$


<img width="838" height="620" alt="image" src="https://github.com/user-attachments/assets/503db83d-309d-483c-808e-92b486baae08" />


#### **Fitting the Curve (Using `numpy.polyfit` and `numpy.poly1d`)**

NumPy provides powerful tools for polynomial fitting.

**Code Example:**

```python
# Create non-linear data
X_poly = np.linspace(0, 4, 20)
Y_poly = 2 * X_poly**2 - 5 * X_poly + 3 + np.random.normal(0, 1, 20)

# Fit a 2nd-degree (quadratic) polynomial
degree = 2
coefficients = np.polyfit(X_poly, Y_poly, degree)
model = np.poly1d(coefficients)

# Calculate R-squared (using the variance formula)
Y_mean = np.mean(Y_poly)
SST = np.sum((Y_poly - Y_mean)**2)  # Total Sum of Squares
SSR = np.sum((model(X_poly) - Y_mean)**2) # Regression Sum of Squares
R_squared_poly = SSR / SST

# Plotting
plt.figure(figsize=(8, 5))
plt.scatter(X_poly, Y_poly, color='green', label='Actual Data')
plt.plot(X_poly, model(X_poly), color='purple', label=f'Polynomial (Degree {degree})\n(R-squared: {R_squared_poly:.3f})')
plt.title('Polynomial Regression (Degree 2)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()
```

**Code Explanation:**

  * `np.polyfit(X, Y, degree)`: Calculates the coefficients of the polynomial that best fits the data in a least-squares sense.
  * `np.poly1d(coefficients)`: Creates a convenient function object based on the coefficients, allowing you to easily calculate the predicted $Y$ values (`model(X_poly)`).

### 3\. Multiple Regression

Multiple Regression is an extension of linear regression that uses two or more independent variables ($X_1, X_2, \dots, X_n$) to predict the dependent variable ($Y$).

$$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_n X_n + \epsilon$$

While Python's standard libraries like `scipy` and `numpy` can handle the underlying matrix operations, this type of modeling is typically done using **Pandas** and specialized libraries like **`statsmodels`** or **`scikit-learn`** because they provide the necessary structured input/output (DataFrames) and statistical summary tables.

### Regression Metrics and Diagnostics

#### **Coefficient of Determination ($R^2$ or R-squared)**

$R^2$ is a statistical measure that represents the proportion of the variance for a dependent variable that's explained by the independent variables in a regression model.

$$\mathbf{R^2 = \frac{SSR}{SST} = 1 - \frac{SSE}{SST}}$$

  * **SST (Total Sum of Squares):** The variance in the data without considering regression. $\sum (Y_i - \bar{Y})^2$
  * **SSE (Error Sum of Squares):** The unexplained variance (residual error). $\sum (Y_i - \hat{Y}_i)^2$
  * **SSR (Regression Sum of Squares):** The variance explained by the model. $\sum (\hat{Y}_i - \bar{Y})^2$
  * $R^2$ ranges from 0 to 1. A value closer to 1 indicates that the model fits the data very well.

<img width="1280" height="720" alt="image" src="https://github.com/user-attachments/assets/0671740b-6bd4-4cc1-8103-ea0d916c8693" />


#### **Residual Error**

The residual ($\epsilon_i$) is the difference between the actual observed value ($Y_i$) and the predicted value ($\hat{Y}_i$) from the regression line.

$$\mathbf{\text{Residual}_i = Y_i - \hat{Y}_i}$$


<img width="734" height="442" alt="image" src="https://github.com/user-attachments/assets/61e72cbd-791d-443e-b432-78c43549929a" />


Analyzing residuals (e.g., plotting them against the predicted values) is critical for validating the assumptions of linear regression, such as homoscedasticity (constant variance of errors).

**Code Example (Calculating and Plotting Residuals):**

```python
# Re-use Simple Linear Regression data and model from Section 1

# Calculate residuals
residuals = Y - Y_pred

# Plotting Residuals
plt.figure(figsize=(8, 5))
plt.scatter(X, residuals, color='darkorange', label='Residuals')
plt.axhline(y=0, color='gray', linestyle='--', label='Zero Error Line')
plt.title('Residual Plot')
plt.xlabel('Hours Studied (X)')
plt.ylabel('Residual Error (Actual Y - Predicted Y)')
plt.legend()
plt.grid(True)
plt.show()
```

**Code Explanation:**

  * `residuals = Y - Y_pred`: Directly calculates the error for each data point.
  * The plot shows the errors scattered around the zero line. For a good linear model, the residuals should be randomly scattered, showing no clear pattern (which is the case here).

-----

## 🌳 Classification Algorithms: Predicting Categories

Classification is a Supervised Learning task where the goal is to predict a discrete, categorical label (or class) for a given input data point.

### 1\. Decision Tree 

A Decision Tree is a flowchart-like structure where the data is recursively split based on features. The algorithm chooses the splits that result in the highest information gain or the lowest impurity.

<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/7d350764-6e4f-4005-b7af-cf4a286ed78a" />



#### **Core Concept: Gini Impurity**

Gini Impurity is a metric used to measure how often a randomly chosen element from a set would be incorrectly labeled if it were randomly labeled according to the distribution of labels in the subset.

  * A Gini Impurity score of **0** means the node is **pure** (all samples belong to the same class).
  * A Gini Impurity score close to 1 means the samples are highly mixed (**high impurity**).

The algorithm seeks to maximize **Information Gain** by minimizing the Gini Impurity in the child nodes after a split.

**Gini Impurity Formula:**
$$I_G(p) = 1 - \sum_{i=1}^{C} (p_i)^2$$
Where $p_i$ is the probability that a sample belongs to class $i$, and $C$ is the number of classes.

#### **Code Example (Decision Tree)**

```python
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Sample Data (X1: Age, X2: Income, Y: Purchase [0=No, 1=Yes])
data = {'Age': [25, 35, 45, 20, 50, 60], 
        'Income': [40, 60, 80, 30, 90, 100], 
        'Purchase': [0, 1, 1, 0, 1, 0]}
df = pd.DataFrame(data)

X = df[['Age', 'Income']]
y = df['Purchase']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Train the Decision Tree model
dt_model = DecisionTreeClassifier(criterion='gini', random_state=42)
dt_model.fit(X_train, y_train)

# 3. Predict and evaluate
y_pred = dt_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Decision Tree Prediction (Test):\n{y_pred}")
print(f"Actual Test Labels:\n{y_test.values}")
print(f"Decision Tree Accuracy: {accuracy:.2f}")
```

**Output Interpretation:**
The output shows the model's prediction on a small test set. **Accuracy** is the fraction of correct predictions ($\text{True Positives} + \text{True Negatives}$ divided by the total number of samples). An accuracy of 1.00 (or 100%) means the model predicted the test labels perfectly.

### 2\. Random Forest

Random Forest is an **ensemble learning** method built on top of Decision Trees. It mitigates the overfitting tendency of a single Decision Tree by constructing multiple trees and averaging their results.


<img width="1358" height="836" alt="image" src="https://github.com/user-attachments/assets/8ba8d58f-46c5-49e9-a43a-3a6c5b25fd8f" />


#### **Core Concept: Majority Voting (Bagging)**

During prediction, every individual Decision Tree in the forest makes a prediction.

  * For classification, the final output is the class that receives the **most votes** (the mode) from all the trees.
  * This averaging process stabilizes the model and significantly improves generalization.

#### **Code Example (Random Forest)**

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Use the same data (X, y) and split (X_train, X_test, y_train, y_test) as used previously

# 1. Train the Random Forest model
# n_estimators=100 means 100 individual decision trees will be built
rf_model = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=42)
rf_model.fit(X_train, y_train)

# 2. Predict and evaluate
y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

print(f"\nRandom Forest Prediction (Test):\n{y_pred_rf}")
print(f"Actual Test Labels:\n{y_test.values}")
print(f"Random Forest Accuracy (100 trees): {accuracy_rf:.2f}")
```

**Output Interpretation:**
The `RandomForestClassifier` trains multiple trees (`n_estimators`) independently and aggregates their results. Similar to the Decision Tree and KNN examples, the accuracy score indicates the model's performance on the unseen test data. Random Forests are highly favored in practice due to their strong predictive power and reduced need for extensive hyperparameter tuning.


### 3\. K-Nearest Neighbors (KNN)

KNN is a non-parametric, **lazy learning** algorithm that uses local information. It memorizes the data during training and only performs computations when a prediction is requested.

<img width="628" height="281" alt="image" src="https://github.com/user-attachments/assets/0e2928cf-64bc-48c3-8e08-eee67d41aacb" />


#### **Core Concepts: $n$-Neighbors ($K$) and Distance Measures**

1.  **Distance Measures:** KNN calculates the distance between the new data point and all existing data points to find the $K$ closest neighbors.
      * **Euclidean Distance** is the most common (straight-line distance).
      * **Manhattan Distance** (city block distance) is another popular choice.
2.  **$n$-Neighbors ($K$):** This hyperparameter determines the size of the neighborhood to examine.
      * The new point is assigned the class label most frequent among its $K$ nearest neighbors (Majority Voting).

The optimal choice of $K$ is crucial: a small $K$ leads to high variance and sensitivity to noise (overfitting), while a large $K$ leads to high bias (underfitting).

**Euclidean Distance Formula (in 2D space):**
$$d(p, q) = \sqrt{(q_1 - p_1)^2 + (q_2 - p_2)^2}$$

#### **Code Example (K-Nearest Neighbors)**

```python
from sklearn.neighbors import KNeighborsClassifier

# Use the same data (X, y) and split (X_train, X_test, y_train, y_test) as above

# 1. Train the KNN model
k_neighbors = 3 # Hyperparameter K
knn_model = KNeighborsClassifier(n_neighbors=k_neighbors, metric='euclidean')
knn_model.fit(X_train, y_train)

# 2. Predict and evaluate
y_pred_knn = knn_model.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)

print(f"\nKNN (K={k_neighbors}) Prediction (Test):\n{y_pred_knn}")
print(f"KNN Accuracy: {accuracy_knn:.2f}")
```

**Output Interpretation:**
This demonstrates how to initialize the `KNeighborsClassifier` by specifying $K$ (here, 3) and the distance metric. The resulting accuracy shows how well the model generalized to the unseen test data.

-----

## 🌌 Unsupervised Learning: Clustering Algorithms

Clustering is an **Unsupervised Learning** task that involves grouping data points such that points within the same group (cluster) are more similar to each other than to those in other groups. Since the data is unlabeled, the algorithm must discover the structure and inherent groupings.

### 1\. K-Means Clustering

K-Means is one of the simplest and most popular clustering algorithms. It partitions $n$ observations into $K$ clusters, where $K$ is specified beforehand.

![k-means](https://dashee87.github.io/images/kmeans.gif)

#### **Core Concepts: Centroids and Initialization**

1.  **Centroids:** The center (mean position) of each cluster. The algorithm aims to minimize the **within-cluster sum of squares (WCSS)**, also known as **inertia**.
2.  **Algorithm Steps:**
      * **Initialization:** Select $K$ random initial centroids (often using **K-Means++** to select centers that are far apart, preventing poor initial groupings).
      * **Assignment (E-Step):** Assign every data point to the nearest centroid based on Euclidean distance.
      * **Update (M-Step):** Recalculate the centroids by taking the mean of all points assigned to that cluster.
      * **Iteration:** Repeat the assignment and update steps until the centroids no longer move significantly (convergence).

**Inertia (WCSS) Formula:**
$$\text{Inertia} = \sum_{j=1}^{K} \sum_{i \in S_j} \|x_i - \mu_j\|^2$$
Where $S_j$ is the $j^{th}$ cluster, $x_i$ is a data point, and $\mu_j$ is the centroid of cluster $j$.

#### **Determining K: The Elbow Method**

Since $K$ must be predefined, the **Elbow Method** is a common technique used to find the optimal number of clusters.

  * The method plots the **Inertia (WCSS)** against the number of clusters ($K$).
  * As $K$ increases, the inertia decreases (more clusters mean points are closer to their centroid).
  * The "elbow" point, where the rate of decrease slows down dramatically, suggests the optimal value of $K$.

#### **Code Example (K-Means Clustering and Elbow Method)**

```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# 1. Create synthetic data (unlabeled)
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# 2. Apply Elbow Method to find optimal K
inertia_values = []
K_range = range(1, 11)

for k in K_range:
    # Set n_init='auto' to silence warnings in recent sklearn versions
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto') 
    kmeans.fit(X)
    inertia_values.append(kmeans.inertia_)

# 3. Plot the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia_values, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia (WCSS)')
plt.grid(True)
plt.show()

# Based on the plot (where the decrease rate slows), K=4 is likely optimal for this data

# 4. Final K-Means Model
optimal_k = 4
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
y_kmeans = kmeans_final.fit_predict(X)

print(f"Data points clustered into {optimal_k} groups.")
print(f"Sample Cluster Labels: {y_kmeans[:10]}")
```

**Output Interpretation:**
The Elbow Plot visually guides the choice of $K$. The final model output (`y_kmeans`) shows which cluster label (0, 1, 2, or 3) was assigned to the first 10 data points.

### 2\. Hierarchical Clustering

Hierarchical Clustering builds a hierarchy of clusters, represented by a tree-like diagram called a **dendrogram**. It does not require specifying the number of clusters ($K$) beforehand.


![Hierarchial-clustering](https://dashee87.github.io/images/hierarch.gif)

#### **Core Concepts: Dendrogram and Linkage**

1.  **Dendrogram:** A tree diagram showing the sequence of merges or splits. The final clusters can be determined by cutting the dendrogram at a specific height.
2.  **Agglomerative (Bottom-Up):** The most common form. It starts with every data point as its own cluster. It then iteratively merges the two closest clusters until only one cluster remains.
3.  **Divisive (Top-Down):** Starts with one cluster (all points) and recursively splits the clusters until every observation is in its own cluster.
4.  **Linkage:** Defines how the distance between two clusters is measured:
      * **Ward Linkage (most common):** Minimizes the variance within each of the merged clusters.
      * **Single Linkage:** Uses the shortest distance between any point in the two clusters.
      * **Complete Linkage:** Uses the maximum distance between any point in the two clusters.

#### **Code Example (Hierarchical Clustering and Dendrogram)**

```python
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
import numpy as np

# 1. Sample Data (features)
data_hierarchical = np.array([[1, 1], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])

# 2. Scaling is important for distance-based methods
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data_hierarchical)

# 3. Perform Agglomerative Linkage
linked = linkage(X_scaled, method='ward')

# 4. Plot the Dendrogram
plt.figure(figsize=(10, 6))
dendrogram(linked, orientation='top', labels=[f"Point {i+1}" for i in range(len(data_hierarchical))])
plt.title('Hierarchical Clustering Dendrogram (Ward Linkage)')
plt.xlabel('Data Points')
plt.ylabel('Distance')
plt.show()
```

**Output Interpretation:**
The Dendrogram shows the merging history. If you cut the dendrogram horizontally (e.g., at a distance of 3), you can see how many clusters are formed at that level. The $y$-axis represents the distance at which clusters were merged.

-----

