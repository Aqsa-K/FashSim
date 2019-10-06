# FashSim
An experimental project to find similarity between clothes based on their patterns and stitching style

Pakistani clothes are rich in patterns but they can still be put into distinct categories. This is an experimetnal project, speicifically for finding similarities in patetrns of pakistani clothes. 

The concept is inspired by Google SMILY project done for medical images; where patches are created and stored and KNN is used to fetch most similar patches given an input patch. 

This repo looks at something similar i.e patches for images are stored and KNN is used to fetch top 'n' patches given an input patch. 

Because we define different patterns as different styles, and style can be measured using Gram Matrix, this additional concept combined with cosine similarity is also used to find similar patches givena n input patch. 
