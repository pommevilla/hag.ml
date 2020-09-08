def evaluate_model(model, test_features, test_labels, model_name):
    import numpy as np

    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model performance: {}'.format(model_name))
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

    return accuracy

def prep_data():
    import pandas as pd
    features = pd.read_csv('data/temps_extended.csv')
    features = pd.get_dummies(features)
    labels = features['actual']
    features = features.drop('actual', axis=1)
    important_features_names = ['temp_1', 'average', 'ws_1',
                                'temp_2', 'friend', 'year']
    feature_list = important_features_names[:]

    features = features[important_features_names]

    import numpy as np

    features = np.array(features)
    labels = np.array(labels)

    from sklearn.model_selection import train_test_split

    train_features, test_features, train_labels, test_labels = train_test_split(features,
                                                                                labels,
                                                                                test_size=0.25,
                                                                                random_state=489)

    return train_features, test_features, train_labels, test_labels


def main():
    import time
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
    from pprint import pprint
    import numpy as np

    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]

    random_grid = {
        'n_estimators': n_estimators,
        'max_features': max_features,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'bootstrap': bootstrap,
    }

    # print(random_grid)
    # pprint(random_grid)

    rf = RandomForestRegressor(random_state=489)

    rf_random = RandomizedSearchCV(estimator=rf,
                                   param_distributions=random_grid,
                                   n_iter=50,
                                   cv=3,
                                   random_state=489,
                                   n_jobs=-1
                                   )

    x_train, x_test, y_train, y_test = prep_data()

    t_0 = time.time()
    rf_random.fit(x_train, y_train)
    t_1 = time.time()

    print('Random search model fitting time: {}'.format(t_1 - t_0))

    base_model = RandomForestRegressor(n_estimators=10,
                                       random_state=42)


    t_0 = time.time()
    base_model.fit(x_train, y_train)
    t_1 = time.time()

    print('Base model fitting time: {}'.format(t_1 - t_0))

    base_accuracy = evaluate_model(base_model, x_test, y_test, "Base model")

    best_random = rf_random.best_estimator_
    random_accuracy = evaluate_model(best_random, x_test, y_test, "Random search CV")

    random_improvement = 100 * (random_accuracy - base_accuracy) / base_accuracy
    print('Improvement of {:0.2f}% over base model.\n'.format(random_improvement))

    param_grid = {
        'bootstrap': [True],
        'max_depth': [80, 90, 100, 110],
        'max_features': [2, 3],
        'min_samples_leaf': [3, 4, 5],
        'min_samples_split': [8, 10, 12],
        'n_estimators': [100, 200, 300, 1000]
    }

    rf = RandomForestRegressor()

    grid_search = GridSearchCV(estimator=rf,
                               param_grid=param_grid,
                               cv=3,
                               n_jobs=-1)

    t_0 = time.time()
    grid_search.fit(x_train, y_train)
    t_1 = time.time()

    print('Grid search fitting time: {}'.format(t_1 - t_0))


    grid_search.best_params_

    best_grid = grid_search.best_estimator_
    grid_accuracy = evaluate_model(best_grid, x_test, y_test, "Grid search")

    grid_improvement = 100 * (grid_accuracy - base_accuracy) / base_accuracy
    print('Improvement of {:0.2f}% over base model.\n'.format(grid_improvement))

    print('Best parameters found:')
    pprint(grid_search.best_params_)


    return 1


if __name__ == '__main__':
    import sys

    sys.exit(main())
