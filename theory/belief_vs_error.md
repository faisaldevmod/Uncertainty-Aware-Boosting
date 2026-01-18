## Belief vs Error

Error answers the question:
"How wrong is the model?"

Belief answers a different question:
"How sure was the model while being wrong?"

Boosting algorithms only see error magnitude.
They do not see belief.

This conflation causes uncertainty to be punished as if it were
confident ignorance, leading to unnecessary overcorrection.

Learning systems should treat these two cases differently.
