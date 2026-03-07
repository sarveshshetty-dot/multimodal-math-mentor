# Probability

## Basic Definitions
Probability of an event $A$:
$P(A) = \frac{\text{Number of favorable outcomes}}{\text{Total number of possible outcomes}}$

## Addition Theorem
For any two events A and B:
$P(A \cup B) = P(A) + P(B) - P(A \cap B)$
If events are mutually exclusive: $P(A \cap B) = 0$

## Conditional Probability
The probability of observing event A given that B has occurred:
$P(A|B) = \frac{P(A \cap B)}{P(B)}$

## Multiplication Rule
$P(A \cap B) = P(A) \cdot P(B|A)$
If events are independent: $P(A \cap B) = P(A)P(B)$

## Bayes' Theorem
$$P(A_i | B) = \frac{P(B | A_i)P(A_i)}{\sum_{k} P(B|A_k)P(A_k)}$$
