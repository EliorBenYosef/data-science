"""
https://www.johnwittenauer.net/machine-learning-exercises-in-python-part-8/

Recommendation Systems
Recommendation engines use item and user-based similarity measures to examine a user's historical preferences
  to make recommendations for new "things" the user might be interested in.

Collaborative Filtering is a particular recommendation algorithm

Here we:
implement collaborative filtering and apply it to a dataset of movie ratings.
build a recommendation system using collaborative filtering and apply it to a movie recommendations dataset.

"""

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# TODO: remove num_features?
def cost(params, Y, R, n_features, learning_rate=None):
    """
    a cost function for collaborative filtering.
    Intuitively, the "cost" is the degree to which a set of movie rating predictions deviate from the true predictions.
    The cost equation is based on two sets of parameter matrices called X and Theta.
    These are "unrolled" into the "params" input so that we can use SciPy's optimization package later on.
    Our next step is to add regularization to both the cost and gradient calculations. We'll create one final regularized version of the function (note that this version includes an additional learning rate parameter called "lambda").

    n_movies = 1682
    n_users = 943
    n_features = 10

    Y - (n_movies, n_users)
    R - (n_movies, n_users)
    """
    n_movies = Y.shape[0]
    n_users = Y.shape[1]

    # reshape params into X & Theta back again
    X = params[:n_movies * n_features].reshape((n_movies, n_features))  # (n_movies, n_features)
    Theta = params[n_movies * n_features:].reshape((n_users, n_features))  # (n_users, n_features)

    # compute the cost
    error = (np.dot(X, Theta.T) - Y) * R  # (n_movies, n_users)
    squared_error = np.power(error, 2)  # (n_movies, n_users)
    J = (1. / 2) * np.sum(squared_error)

    # add the cost regularization
    if learning_rate is not None:
        J += ((learning_rate / 2) * np.sum(np.power(Theta, 2)))
        J += ((learning_rate / 2) * np.sum(np.power(X, 2)))

    # compute the gradients
    X_grad = np.dot(error, Theta)
    Theta_grad = np.dot(error.T, X)

    # add the gradients regularization
    if learning_rate is not None:
        X_grad += (learning_rate * X)
        Theta_grad += (learning_rate * Theta)

    # unravel the gradient matrices into a single array
    grad = np.concatenate((np.ravel(X_grad), np.ravel(Theta_grad)))

    return J, grad


def create_user_movie_rating(n_movies=1682):
    """
    creating our own movie (1-5) ratings so we can use the model to generate personalized recommendations.
    """
    ratings = np.zeros((n_movies, 1))
    ratings[0] = 4
    ratings[6] = 3
    ratings[11] = 5
    ratings[53] = 4
    ratings[63] = 5
    ratings[65] = 3
    ratings[68] = 5
    ratings[97] = 2
    ratings[182] = 4
    ratings[225] = 5
    ratings[354] = 5
    return ratings


def get_movie_indexes():
    """
    A file is provided for us that links the movie index to its title. Let's load the file into a dictionary and use some sample ratings provided in the exercise.
    """
    movie_idx = {}
    f = open('../../../datasets/per_type/txt/movie_ids.txt')
    for line in f:
        tokens = line.split(' ')
        tokens[-1] = tokens[-1][:-1]
        movie_idx[int(tokens[0]) - 1] = ' '.join(tokens[1:])
    return movie_idx


data = loadmat('../../../datasets/per_type/matlab/movies.mat')

# Y - the users' movies ratings - a (number of movies x number of users) array containing ratings from 1 to 5.
Y = data['Y']  # (n_movies, n_users)
# R - an "indicator" array containing binary values indicating if a user has rated a movie or not.
R = data['R']  # (n_movies, n_users)

# the average rating for a movie is - Y[i,R[i,:]].mean()
#   done by averaging over a row in Y for indexes where a rating is present.


# # Visualization of the data by rendering the matrix as if it were an image.
# #   We can't glean too much from this
# #       but it does give us an idea of a relative density of ratings across users and movies.
# fig, ax = plt.subplots(figsize=(12,12))
# ax.imshow(Y)
# ax.set_xlabel('Users')
# ax.set_ylabel('Movies')
# fig.tight_layout()
# plt.show()

# A set of pre-trained parameters that we can evaluate:
params_data = loadmat('../../../datasets/per_type/matlab/movie_params.mat')
X = params_data['X']
Theta = params_data['Theta']

####################################

# Testing the cost function
#   using only a small sub-set of the data, to keep the evaluation time down.
n_users = 4
n_movies = 5
n_features = 3

Y_sub = Y[:n_movies, :n_users]
R_sub = R[:n_movies, :n_users]

X_sub = X[:n_movies, :n_features]
Theta_sub = Theta[:n_users, :n_features]
params = np.concatenate((np.ravel(X_sub), np.ravel(Theta_sub)))  # params has to be 1D!

J, grad = cost(params, Y_sub, R_sub, n_features, learning_rate=1.5)  # learning_rate=None
print('Initial cost:', J)  # 22.224603725685675 \ 31.34405624427422 with regularization

####################################

# create user ratings and add to data:
#   add this custom ratings vector to the dataset so it gets included in the model.
ratings = create_user_movie_rating()
Y = np.append(Y, ratings, axis=1)
R = np.append(R, ratings != 0, axis=1)


# training the collaborative filtering model
# We're going to normalize the ratings and then run the optimization routine
#   using our cost function, parameter vector, and data matrices at inputs.
n_movies = Y.shape[0]
n_users = Y.shape[1]
n_features = 10
learning_rate = 10.

X = np.random.random(size=(n_movies, n_features))
Theta = np.random.random(size=(n_users, n_features))
params = np.concatenate((np.ravel(X), np.ravel(Theta)))  # params has to be 1D!

Y_mean = np.zeros((n_movies, 1))
Y_norm = np.zeros((n_movies, n_users))
for i in range(n_movies):
    idx = np.where(R[i,:] == 1)[0]
    Y_mean[i] = Y[i, idx].mean()
    Y_norm[i, idx] = Y[i, idx] - Y_mean[i]

fmin = minimize(fun=cost, x0=params, args=(Y_norm, R, n_features, learning_rate),
                method='CG', jac=True, options={'maxiter': 100})


# Since everything was "unrolled" for the optimization routine to work properly,
#   we need to reshape our matrices back to their original dimensions.
X = fmin.x[:n_movies * n_features].reshape((n_movies, n_features))  # (n_movies, n_features)
Theta = fmin.x[n_movies * n_features:].reshape((n_users, n_features))  # (n_users, n_features)


# Our trained parameters are now in X and Theta.
#   We can use these to create some recommendations for the user we added earlier.
predictions = np.dot(X, Theta.T)
my_preds = predictions[:, np.newaxis, -1] + Y_mean
# That gives us an ordered list of the top ratings, but we lost what index those ratings are for.
sorted_preds = np.sort(my_preds, axis=0)[::-1]
# We actually need to use argsort so we know what movie the predicted rating corresponds to.
idx = np.argsort(my_preds, axis=0)[::-1]


# print predictions
print("Top 10 movies' predicted ratings:")  # rating predictions
movie_idx = get_movie_indexes()
for i in range(10):
    j = int(idx[i])
    print(f'{my_preds[j][0]:.2f} - {movie_idx[j]}')

# The recommended movies don't actually line up that well with what's in the exercise text.
# The reason why isn't too clear and I haven't found anything to account for it,
#   but it's possible there's a mistake in the code somewhere.
# Still, even if there's some minor difference the bulk of the example is accurate.
