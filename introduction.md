
# A Vehicle Tutorial 

## Outline

0. Introduction and Motivation for Vehicle

1. A simple semantically meaningful example.
    - Maybe hierarchical classification? e.g. In this dogs dataset it shouldn't confuse an Afghan hound and a Border terrier.
    - Introduce basic syntax and properties.
2. A more complicated semantically meaningful example 
    - e.g. AcasXu
    - Introduces building and reusing functions.
3. Less semantically meaningful
    - e.g. MNIST robustness
    - Introduces concepts of datasets, parameters etc.
4. Semantically meaningful + integration with Agda.
    - Braking example?
    - Vehicle controller?

## Introduction and Motivation
 
### What is Neural Network Verification about?

Neural networks are widely used in the field of machine learning; and are often embedded as *pattern-recognition* or *signal processing* components into complex software. In some scenarious, it becomes important to establish formal guarantees about their behaviour. Following the pioneering work of [@Katz2017,Singh2019,Wang2021] neural network verification has become an active research area. 

Formally, a neural network is a function $N : R^m \rightarrow R^n$. Verification of such functions most commonly boils down to specifying admissible intervals for the function's output given an interval for its inputs. For example, one can specify a set of inputs to belong to an $\epsilon$- neighborhood of some given input $\mathbf{x}$, and verify that for such inputs, the outputs of $N$ will be in $\delta$ distance to $N(\mathbf{x})$. This property is often called $\epsilon$*-ball robustness* (or just *robustness*), as it proves the network's output is robust (does not change drastically) in the neighborhood of certain inputs.

Seen as functions, neural networks have two particular features that play an important role in their verification: these functions are not written manually, but generated (or *fitted*) to model the given data distribution. As a consequence, "big data" often requires one to use large neural networks, and we often attribute very little semantic or structural meaning to the resulting function. 

### Challenges in Neural Network Verification

There are four main research challenges in this area: 
1. Scalability of (semi-)decision procedures that check the property satisfaction for neural networks. State-of-the art neural network verifiers [...], based on a combination of abstract interpretation algorithms and domain-specific heuristics can verify neural networks of size ... whereas large industrial models used by Amazon or Google reach the size of .... (I need help here from Matthew, Marco or Natalia)
2. Limited scope of neural network properties available in the literature. Arguably,  $\epsilon$-ball robustness has very limited practical applications. Various efforts of the community to broaden the range of verifiable properties mainly resulted in domain specific solutions (see eg [] for verification of networks used in air collision avoidance). 
3. Neural networks are rarely used as stand-alone oracles. They are usually part of more complex systems. Verifying the network's behavior within a larger system is an area that still requires investigation. (maybe good to give a few citations here for existing work)
4. Because a given neural network is generated to fit the data, rather than to satisfy a given property, in the majority of cases, a naive attempt to verify the network results in failure to establish that the property actually holds.  For example, as reported in [], a 99% accurate network may only be proven robust in the neighborhood of 1% of its images. However, one can re-train the network by translating a given property into a loss function, and this can dramatically increase the chances that the network satisfies the property: [] shows in the best case an increase from 1% to 90%. (Check DL2 paper and last year's stats from Marco's paper and give citations).    

Combined together, these four points make practical use of verification techniques inaccessible for the majority of machine learning practitioners and even researchers, despite of the high demand for safety guarantees in complex intelligent systems. Imagine, for example, a designer of a chatbot who tries to prove that the chatbot does not offend or mislead a human user, and suppose they were lucky enough to install one available neural network verifier.  For a start, their neural network may be too big to verify, so they need to make it small enough. Next, they need to define formally what "offend or mislead" is, the task that is made harder by having to use a relatively low-level language of the verifier that expects the properties to be stated on the level of individual inputs and neurons. If they have made through that hurdle, and finally can run the "verify" command line, they may suddenly find that their property fails for the model they have. So, they need to return to square one and train a different model, that does satisfy the property. Unfortunately, the installed verifier will not be able to help with this task!

### What does Vehicle Team Propose?

In this tutorial, we present the tool Vehicle that provides support for this complex verification cycle: It provides an environment that allows one to express neural network specifications in a high-level, human-readable format. Then it compiles them into low-level queries that can be passed automatically to verifiers to prove whether the specification holds or provide a counterexample. If the specification cannot be verified, Vehicle gives one an option to automatically generate a new loss function that can be used to train the model to satisfy the stated property. 
Once a specification has been verified (possibly after property-driven re-training), Vehicle allows one to export the proof to an interactive theorem prover, and reason about the behavior of the complex system that embeds the machine learning model. 

Vehicle programs can be compiled to an unusually broad set of backends,
including: 

 a) loss functions for Tensorflow which can be used to guide 
 both specification-directed training and gradient-based counter-example
 search.
 
 b) queries for the Marabou neural network verifier, which
 can be used to formally prove that the network obeys the specification.
 
 c) Agda specifications, which are tightly coupled to the original network
 and verification result, in order to scalably and maintainably construct
 larger proofs about machine learning-enhanced systems.
 
Currently, Vehicle supports the verifier Marabou, the ITP Agda, and the ONNX format for neural networks.

### Objectives of this Tutorial

This tutorial will give an introduction to the Vehicle tool 
(https://github.com/vehicle-lang/vehicle) and its conceptual approach
to modelling specifications for machine learning systems via functional
programming. It will teach the participants to understand the 
range of problems that arise in neural network property specification, 
verification and training, and will give a hands-on experience on 
solving these problems at a level of a higher-order specification 
language with dependent types.

### Prerequisites

To follow the tutorial, you will need Vehicle, Marabou and Agda installed in your machine.
For instructions, refer to [vehicle documentation](https://vehicle-lang.readthedocs.io/en/latest/installation.html).
You can also download already trained networks for our examples from [link to tutorial repo].

(Recommendation to use vsc with vcl syntax highlighting)


### Related work

- Guy Katz, Clarke Barrett, D. Dill, K. Julian, and M. Kochenderfer. Reluplex: An Efficient SMT
Solver for Verifying Deep Neural Networks. In CAV, 2017.
- Gagandeep Singh, Timon Gehr, Markus Püschel, and Martin T. Vechev. An abstract
domain for certifying neural networks. Proc. ACM Program. Lang., 3(POPL):41:1–41:30, 2019.
- Shiqi Wang, Huan Zhang, Kaidi Xu, Xue Lin, Suman Jana, Cho-Jui Hsieh, and J. Zico
Kolter. Beta-crown: Efficient bound propagation with per-neuron split constraints for neu-
ral network robustness verification. In Marc’Aurelio Ranzato, Alina Beygelzimer, Yann N.
Dauphin, Percy Liang, and Jennifer Wortman Vaughan, editors, Advances in Neural In-
formation Processing Systems 34: Annual Conference on Neural Information Processing
Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual, pages 29909–29921, 2021.

## Vehicle Preliminaries

- introduction of dataset and models
- introduction of basic syntax
- describe a property in words
- example with NN?
- example of property not holding 
- fix in vehicle!
