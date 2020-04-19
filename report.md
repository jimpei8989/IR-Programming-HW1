# Web Retrieval & Web Mining<br>Programming HW1

###### By: Wu-Jun Pei (B06902029)

### § Vector Space Model

The vector space model I use in this programming homework is **<u>Okapi BM25</u>**, the model the professor suggests.

- Term Frequency for term $t$, document $d$
    $$
    TF(t, d) = \frac{c(t, d) \cdot (k_1 + 1)}{c(t, d) + k_1 \cdot (1 - b + b \cdot \frac{|d|}{\text{avgDocLen}})}
    $$
    , where $c(t, d)$ is the frequency count for term $t$ in document $d$.

- Inverse Document Frequency for term $t$
    $$
    IDF(t) = \log \frac{N - n(t) + 0.5}{n(t) + 0.5}
    $$
    , where $n(t)$ is the number of documents containing $t$.
    
- Score for term $t$ and document $d$
    $$
    Score(t, d) = TF(t, d) \cdot IDF(t)
    $$

Following the suggestions on wikipedia, I set $b = 0.75$ and tried $k_1 \in \{1.2, 1.6, 2.0\}$. At last, I choose $k_1 = 2.0$ since it performs the best among the three.

##### Reference

- [Wikipedia - Okapi BM25](https://en.wikipedia.org/wiki/Okapi_BM25)
- Lecture slides

### § Relevance Feedback

After taking a look at `ans_train.csv`, I found that those answers is not really helpful to our testing set since the intersection between them is small. Thus, I simply implement <u>**Pseudo Relevance Feedback**</u>.

##### Implementation

Take the top $K$ retrieved documents in the first round, calculate the centroid $\vec C_K$ and add it back to our query vectors.
$$
\vec q = \alpha \vec q_0 + \beta \vec C_K
$$
- Normalize both $\vec q_0$ and $\vec C_K$ before adding them together.
- Choose $K \in \{10, 15\}$, $\beta \in \{0.1, 0.2, 0.3\}$
- Try to do relevance feedback for several iterations.

### § Results of Experiments

##### Result Table

|  ID  |            Name             |     $k_1$      |       Relevance<br>Feedback       |  Training<br/>MAP  |   Public<br/>MAP   | Private<br/>MAP |
| :--: | :-------------------------: | :------------: | :-------------------------------: | :----------------: | :----------------: | :-------------: |
|  1   |      `OkapiBM25-0xa0`       |      1.2       |               False               |      0.67366       |      0.76497       |                 |
|  2   |      `OkapiBM25-0xa1`       |      1.6       |               False               |      0.67401       |      0.76815       |                 |
|  3   | <u>**`OkapiBM25-0xa2`**</u> | <u>**2.0**</u> |         <u>**False**</u>          | <u>**0.67514**</u> | <u>**0.76994**</u> |                 |
|  4   |      `OkapiBM25-0xa3`       |      2.4       |               False               |      0.67273       |        TODO        |                 |
|  5   |      `OkapiBM25-0xa2`       |      2.0       |          K = 10, ß = 0.1          |      0.66396       |      0.77130       |                 |
|  6   |      `OkapiBM25-0xa2`       |      2.0       |          K = 15, ß = 0.1          |      0.66396       |        TODO        |                 |
|  7   | <u>**`OkapiBM25-0xa2`**</u> | <u>**2.0**</u> |    <u>**K = 10, ß = 0.2**</u>     | <u>**0.65258**</u> | <u>**0.77187**</u> |                 |
|  8   |      `OkapiBM25-0xa2`       |      2.0       |          K = 15, ß = 0.2          |      0.65258       |        TODO        |                 |
|  9   |      `OkapiBM25-0xa2`       |      2.0       |          K = 10, ß = 0.3          |      0.63628       |        TODO        |                 |
|  10  |      `OkapiBM25-0xa2`       |      2.0       | K = 10, ß = 0.05<br>5 iterations  |      0.64840       |        TODO        |                 |
|  11  |      `OkapiBM25-0xa2`       |      2.0       | K = 10, ß = 0.05<br>10 iterations |      0.60820       |        TOD         |                 |

Some notes

- The **<u>bold and underline</u>** submissions are the two I selected for final score.
- The training MAP is calculated based on `ans_train.csv`.

##### About Vector Space Model

1. Okapi BM25 is strong enought to pass the strong baseline easily.
2. Tuning the hyperparameter $k_1$ will not make a huge improve.

##### About Relevance Feedback

1. The relevance feedback performs much badly on training set but gains a little improve on public testing set.
2. There is no difference between choosing $K = 10$ and $K = 15$. The reason might be the centroid between them are so similar that the modified query vector won't differ a lot.
3. As for $\beta$, choosing either $0.1$ or $0.2$ will lead to similar result. However, when $\beta = 0.3$, the result becomes terrible.
4. I've also tried to increase the iterations to perform relevance feedback. However, the result turns out to be even worse (compared to the 7-th one).
5. To conclude, I think increasing $K$, $\beta$ or number of iterations blindly will make the output less precise since the query vector has been modified too much.

### § Discussion

In this homework, I learned

1. Implement a VSM model, I found that

