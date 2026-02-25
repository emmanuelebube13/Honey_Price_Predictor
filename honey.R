# 1. Load the tools and the data
library(caret)
csv <- read.csv("honey_purity_dataset.csv")

# 2. The Translator: One-Hot Encode the Pollen_analysis column
dummy_model <- dummyVars(~ ., data = csv)
encoded_csv <- data.frame(predict(dummy_model, newdata = csv))

# 3. The Testing Arena: Set up the 10-Fold Cross-Validation, repeated twice
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 2)

# 4. The Brain: Train the Lasso Model
# First, define the penalty parameters the assignment gave you
parameters <- c(seq(0.1, 2, by = 0.1), seq(2, 5, 0.5), seq(5, 25, 1))

# Now, train the model (assuming we are predicting 'Price')
final_model <- train(
  Price ~ ., 
  data = encoded_csv,
  method = "glmnet",
  trControl = fitControl,
  tuneGrid = expand.grid(alpha = 1, lambda = parameters)
)

# 5. Print the summary for your Brightspace screenshot
print(final_model)