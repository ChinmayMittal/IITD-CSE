The first few rounds were common for both the trading and the software engineering roles. They were nothing very difficult but involved quick calculations.

The first few rounds included simple puzzles, 80 in 8, and other such games which tested quick calculations, memorizing ability, basic logical reasoning among other things. They are relatively easy to crack without any practice.

The next few rounds were different for both roles.

*1st Software Specific Round*
- Coding test in C++. 
- 3 questions
- The questions were based on trading problems. 
- They were not that difficult and did not require any trading knowledge in particular. 
- Some CP/DSA-specific knowledge was required.
- They were a little lengthy, so time management was important

I did not solve all three questions but was still selected not sure what the selection criterion was for interviews, but in my opinion, it was relatively lenient.

*HR Interview*

The first interview was one with the HR. Unlike other companies, Optiver focuses a little more on your personality apart from technical skills. It is important to be confident and clear while speaking.

- We discussed very general things. Nothing to be prepared for as such.
- I remember talking about things like why I want to join Optiver. Life in Amsterdam. Work culture in Optiver and the projects in my CV, and my interests in general.

*System Design Interview*

This was a bit open-ended and different from most interviews. So be prepared.

The problem statement was that given you are a trader in Amsterdam with the exchange in Frankfurt, design a system to allow you to trade.

- They want to test how you think
- Interact with the interviewer. Think loudly !
- Break the problem into parts. Discuss the decisions you make and why you make them
- Ask for clarifications and assumptions, if any, and engage with the interviewer

Eg. 
- We'll start setting some form of communication between the trader and the exchange.
- What kind of data will we send ?
- What is the latency and bandwidth of our data transfer ?
- The trader will have a screen with prices and options to buy and sell, which will be communicated with the exchange
- How do we improve this setup. Identify the bottleneck which is the trader clicking buttons in this case
- Automate the job the trader is doing, decreasing the latency
- The next bottleneck becomes the data transfer from Amsterdam to Frankfurt
- Remove this bottleneck by shifting the algorithm to co-location in Frankfurt.
.....
This way, keep improving coming to a complete solution.

*Trading Software Interview*

It was again a relatively simple coding question involving writing basic pseudocode for a trading problem.

It was not that difficult and was to understand how we think, why we make certain decisions, and how well we communicate our ideas.

Again be confident, and ask for clarifications. Think loudly and explain your decisions.
