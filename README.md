# CSE 151A Milestone 5: Third Model(s)

### **Note: For graphs and code, please see notebook. They are included there.**

Repo link: https://github.com/cse151a-nba-project/milestone-5/
Data link: https://github.com/cse151a-nba-project/data/

Group Member List: 

- Aaron Li all042\@ucsd.edu
- Bryant Jin brjin\@ucsd.edu
- Daniel Ji daji\@ucsd.edu
- David Wang dyw001\@ucsd.edu
- Dhruv Kanetkar dkanetkar\@ucsd.edu
- Eric Ye e2ye\@ucsd.edu
- Kevin Lu k8lu\@ucsd.edu
- Kevin Shen k3shen\@ucsd.edu
- Max Weng maweng\@ucsd.edu
- Roshan Sood rosood\@ucsd.edu

Abstract, for reference: 

Although sports analytics captured national attention only in 2011 with the release of Moneyball, research in the field is nearly a century old. Until relatively recently, this research was largely done by hand; however, the heavily quantitative nature of sports analytics makes it an attractive target for machine learning. This paper explores the application of advanced machine learning models to predict team performance in National Basketball Association (NBA) regular season and playoff games. Several models were trained on a rich dataset spanning 73 years, which includes individual player metrics, opponent-based performance, and team composition. The core of our analysis lies in combining individual player metrics, opponent-based game performances, and team chemistry, extending beyond traditional stat line analysis by incorporating nuanced factors. We employ various machine learning techniques, including neural networks and gradient boosting machines, to generate predictive models for player performance and compare their performance with both each other and traditional predictive models. Our analysis suggests that gradient boosting machines and neural networks significantly outperform other models. Neural networks demonstrate significant effectiveness in handling complex, non-linear data interrelations, while gradient boosting machines excel in identifying intricate predictor interactions. Our findings emphasize the immense potential of sophisticated machine learning techniques in sports analytics and mark a growing shift towards computer-aided and computer-based approaches in sports analytics.


# 1. Evaluate your (Milestone 4) data, labels and loss function. Were they sufficient or did you have have to change them.

Reference: https://github.com/cse151a-nba-project/milestone-4/blob/main/CSE_151A_Milestone_4.ipynb

After evaluating our data, labels, and loss function for Milestone 4, we believe that they were sufficient and well-suited for our SVR (Support Vector Regression) model. We did not have to make any significant changes to them.

Regarding the data, our decision to use the comprehensive NBA player and team performance dataset from 1990-2023 has proven to be effective. This dataset provides a wealth of information that captures the performance of players and teams over a substantial period. We believe that using data from this specific range strikes a good balance between having enough data points to train our model effectively and avoiding the inclusion of unrepresentative data from earlier eras, where the game and player dynamics might have been significantly different. Increasing the year range further into the past could introduce noise and hinder the model's ability to capture relevant patterns, while decreasing the year range might result in insufficient data and potential overfitting.

In terms of labels, our selection of features such as 'ts_percent', 'experience', 'x3p_ar', 'per', 'ws_48', 'usg_percent', 'bpm', and 'vorp' has provided a comprehensive representation of player and team performance. These labels encompass various aspects of the game, including shooting efficiency, player experience, three-point attempt rate, player efficiency rating, win shares per 48 minutes, usage percentage, box plus-minus, and value over replacement player. By incorporating these diverse metrics, we can capture the multi-faceted nature of player contributions and their impact on team success. The combination of these labels has proven to be informative and relevant for predicting team win percentages.

Moreover, our decision to consider the top 10 players in every NBA team has been effective in capturing the core contributors to team performance. By focusing on the top 10 players, we ensure that we are accounting for the most impactful players who receive significant playing time and have a substantial influence on the team's success. This approach allows us to capture the essential information while managing the dimensionality of the data, preventing issues related to sparsity or irrelevant data points.

Regarding the loss function, it is important to note that since we are using an SVR model, we do not rely on the same loss functions typically used in deep neural networks (DNNs) or linear regression models. SVR uses a different optimization objective and employs the concept of margin-based loss, specifically the epsilon-insensitive loss function. This loss function is designed to find a hyperplane that fits the data within a specified margin while allowing for some tolerance for errors. The goal is to minimize the margin violation and find a balance between fitting the training data and generalizing well to unseen data.

Furthermore, the choice of kernel function in SVR, such as the radial basis function (RBF) kernel, allows us to capture non-linear relationships between the features and the target variable. This is particularly useful in our case, as the relationship between player statistics and team win percentages may not always be linear. The kernel function implicitly maps the input features into a higher-dimensional space, enabling the model to find complex patterns and make more accurate predictions.

In conclusion, after thorough evaluation, we believe that our data, labels, and loss function were sufficient and appropriate for our SVR model. The comprehensive NBA player and team performance dataset from 1990-2023, along with the selected labels and the focus on the top 10 players in each team, provides a robust foundation for training our model. The epsilon-insensitive loss function used in SVR is well-suited for our regression task and allows for flexibility in handling noise and outliers. Therefore, we did not find it necessary to make significant changes to our data, labels, or loss function for Milestone 4.

# 2. Train your third model

We decided to train a SVR (Support Vector Regression) model as well as a final Ensemble model (combining all models to predict input data) for this final milestone. We thought this would be a new perspective on the problem since SVR is a powerful and versatile algorithm that can handle non-linear relationships between features and the target variable. Unlike our previous models, such as linear regression and neural networks, SVR has the ability to capture more complex patterns in the data by mapping the input features into a higher-dimensional space using kernel functions. See code, results, and further analysis in the notebook.

# 3. Evaluate your model compare training vs test error

|              | Linear | Elastic Net |  DNN   | Tuned DNN | SVR model | Ensemble model |
|--------------|--------|-------------|--------|-----------|-----------|----------------|
| Training MSE | 16.704 |   17.376    | 19.954 |  11.957   | 12.083    | 14.803         |
| Training MAE | 3.2729 |   3.3292    | 3.5214 |  2.3812   | 2.134     | 2.995          |
| Training R^2 | 0.9312 |   0.9284    | 0.9178 |  0.9507   | 0.9502    | 0.9390         |
| Testing MSE  | 20.881 |   19.921    | 24.686 |  37.071   | 25.281    | 19.634         |
| Testing MAE  | 3.7103 |   3.6713    | 4.1783 |  4.7918   | 4.168     | 3.680          |
| Testing R^2  | 0.9028 |   0.9072    | 0.8850 |  0.8274   | 0.8823    | 0.9086         |

# 4. Where does your model fit in the fitting graph, how does it compare to your first (and best) model?

Based on the performance metrics we obtained and previous models, we can see the SVR model is likely in a spot of overfitting and will likely not improve significantly more with feature expansion. We can see this because the SVR model has training MSE of and testing MSE of 12 and 25 (relatively far apart, with testing greater, suggesting that the model is overfitting), while our best model, the elastic net model (which is likely near the best fit region), has a training MSE of 17 and testing MSE of 20. It seems that we are near a lower boundary of 15 MSE, where models cannot fit below without overfitting, possibly trying to analyze patterns / predict randomness. Even compared to the original linear regression model (which is also likely near the best fit region), our SVR model is worse (a 25 testing MSE vs. 21 testing MSE), indicating that the model complexity is too high and at the overfitting stage. Nevertheless, the R^2 values for both training and testing for the SVR model are above 0.88, indicating a strong correlation between the predicted and actual win percentages. 

For the Ensemble model, testing MSE is 20 and R^2 coefficient is 0.91, which is relatively the same as the elastic net model performance: likely in the best fit region of the fitting graph. Combining all models' predictions together and averaging them results in likely the best model, better or equal than any individual model. Nevertheless, the metrics are about the same as the elastic net model.

# 5. Did you perform hyper parameter tuning? K-fold Cross validation? Feature expansion? What were the results?

Yes, we performed hyperparameter tuning and cross-validation for our SVR model. We did feature expansion for the previous milestone (4) and determined that other stats (from the dataset) were already represented and more likely to cause overfitting than model improvement, so we decided not to do feature expansion this milestone.   

For the SVR model, we used GridSearchCV to search for the best hyperparameters. We defined a parameter grid with different values for the kernel, regularization parameter (C), epsilon, kernel coefficient (gamma), and degree (for polynomial kernel function). We used 5-fold cross-validation (cv=5) to evaluate the model's performance for each combination of hyperparameters. GridSearchCV then selected the best combination of hyperparameters based on the model's performance. After finding the best hyperparameters for the SVR model, we fit the model using the best parameters.

The results of hyperparameter tuning showed that the best parameters for the SVR model were found through GridSearchCV, in this case values of {'C': 0.1, 'degree': 2, 'epsilon': 0.01, 'gamma': 0.01, 'kernel': 'poly'} and score of 29. **Measuring via testing MSE, We observe that the SVR model is better than our DNN models it roughly has the same performance as manual-tuned DNN model, and is not as performant as the elastic net model.**

# 7. Conclusion section: What is the conclusion of your 3rd model? What can be done to possibly improve it? How did it perform compared with your other models and why?

In conclusion, our third model, a SVR model, provided a new way to try to model the NBA data. Although it was not as performant as the elastic net model from the previous milestone, it was nevertheless an interesting exploration. While the elastic net model had test data MSE of 20 and R^2 value of 0.91, the SVR model had test data MSE of 25 and R^2 value of 0.88. However, compared to the DNN models, we generally did better or the same, with lower error and a higher R^2 coeffecient between predicted and mean, indicating a higher correlation (and therefore better prediction) of the two.

Another model part of this milestone was the Ensemble model we created by averaging all the predictions of other models we've created (linear regression, elastic net, DNN models, and SVR model) and returning this averaged result as a prediction. This model was as performant as the elastic net model, or even a slightly better, with testing MSE of 20 and R^2 of 0.91.

To further improve our models, we can (continue to) explore several avenues:

Regularization Techniques: Investigate different regularization techniques for the DNN models, such as L1 and L2 regularization, dropout, or early stopping, to mitigate overfitting and enhance generalization.

Hyperparameter Fine-tuning: Conduct more extensive hyperparameter tuning (with more potential values for each hyperparameter and more SVR hyperparameters being tuned) for the SVR models to find the optimal combination of hyperparameters that minimizes the error metrics.

Feature Selection and Engineering: Revisiting feature importance and considering other potentially relevant features or engineering new features based on domain knowledge to improve the models' predictive power.

Data Augmentation: Collect more diverse and representative data to increase the models' exposure to different scenarios and improve their generalization ability.

Compared to our first linear regression model, this final SVR model did not achieve a significant performance improvement. In fact, it unfortunately overfitted and did worse on testing data. The Elastic Net model from the previous milestone remains our best model. 
