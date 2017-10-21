#English_Text_Analysis_Using_Hidden_Markov_Models

This is a Machine Learning Project which utilizes Hidden Markov Models to determine the structure of English language.

Hidden Markov Models were used for text analysis since a long time.
A HMM consists of three components: 
1. A matrix, also known as the transitional probability matrix
2. B matrix, also known as the observation sequence matrix
3. π matrix, also known as the initial probability distribution vector

So we can construct our HMM as 
			λ = (A, B, π)

All these matrices are row stochastic, which means their probabilities sum to 1
By training the HMM over a series of iterations (Baum-Welsch re-estimation algorithm) and updating the model, certain structures of the elements being trained start to appear in the B matrix. We can understand the structure of the B matrix, by choosing certain parameters, namely N and M.

For example, we have 26 alphabets in the English language (with space, it becomes 27). Suppose someone who does not understand the English language, wants to understand the characters that correspond to consonants, and the ones that correspond to vowels. We know by looking at the above information that
			M = 27 and N = 2
since there are two distinctive states that we want to analyze.
We create an observation sequence of plain English text (with no punctuation, all lowercase, no special symbols), with M = 27 and N = 2 we feed this information to the HMM and after running for certain number of iterations, we are able to make some sense from the structure of the final B matrix, i.e. the structure of English language. In the sense of this example, the B matrix would provide us with information regarding which letters correspond to vowels and which correspond to consonants. By increasing the value of N, we are increasing the dimensions in which we want to analyze English text and the B matrix would provide you with the necessary structural information.

The program hmm.c achieves this by reading from the brown.txt file in order to create the observation sequence and treating it as a input. The structure information would be present in the final B matrix.   
