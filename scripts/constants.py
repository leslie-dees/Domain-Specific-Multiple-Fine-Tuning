# Number of training epochs (iterations over the entire dataset)
EPOCHS = 5

# Size of the batches of data (number of data points considered in a single update step)
BATCH_SIZE = 32

# Learning rate for the optimizer (determines the step size at each iteration while moving toward a minimum of a loss function)
LEARNING_RATE = 2e-5

# Term added to the denominator to improve numerical stability in the optimizer
EPSILON = 1e-8

# Number of warmup steps for learning rate scheduling (gradually increases learning rate to prevent training instability at the start)
WARMUP_STEPS = 0

# Maximum length of tokens (input text will be truncated or padded to this length)
TOKENIZER_MAX_LENGTH = 128
