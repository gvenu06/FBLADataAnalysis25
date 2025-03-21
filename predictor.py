import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as path_effects

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica Neue', 'Helvetica', 'Arial']
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8
plt.rcParams['xtick.color'] = '#333333'
plt.rcParams['ytick.color'] = '#333333'

COLORS = {
    'ozone': ['#1A5276', '#2874A6', '#3498DB', '#5DADE2', '#85C1E9'],
    'no2': ['#641E16', '#922B21', '#C0392B', '#E74C3C', '#F1948A'],
    'pm25': ['#145A32', '#196F3D', '#229954', '#27AE60', '#58D68D'],
    'background': '#F8F9F9'
}


def prepare_data_for_prediction(df, pollutant_name):
    try:
        pollutant_data = df[df['Name'] == pollutant_name].copy()
        print(f"Found {len(pollutant_data)} rows for {pollutant_name}")

        pollutant_data['Start_Date'] = pd.to_datetime(pollutant_data['Start_Date'])
        pollutant_data['Year'] = pollutant_data['Start_Date'].dt.year
        pollutant_data['Month'] = pollutant_data['Start_Date'].dt.month
        pollutant_data['Day'] = pollutant_data['Start_Date'].dt.day
        pollutant_data['DayOfYear'] = pollutant_data['Start_Date'].dt.dayofyear
        pollutant_data['Season'] = pollutant_data['Start_Date'].dt.month.map(
            lambda x: 'Winter' if x in [12, 1, 2] else
            'Spring' if x in [3, 4, 5] else
            'Summer' if x in [6, 7, 8] else 'Fall'
        )

        pollutant_data['Month_sin'] = np.sin(2 * np.pi * pollutant_data['Month'] / 12)
        pollutant_data['Month_cos'] = np.cos(2 * np.pi * pollutant_data['Month'] / 12)

        pollutant_data['Prev_Month_Value'] = np.nan
        pollutant_data['Prev_Year_Value'] = np.nan
        pollutant_data['Rolling_3M_Avg'] = np.nan
        pollutant_data['Rolling_12M_Avg'] = np.nan

        pollutant_data = pollutant_data.sort_values('Start_Date')

        for location in pollutant_data['Geo Place Name'].unique():
            location_mask = pollutant_data['Geo Place Name'] == location
            pollutant_data.loc[location_mask, 'Prev_Month_Value'] = pollutant_data.loc[
                location_mask, 'Data Value'].shift(1)
            pollutant_data.loc[location_mask, 'Prev_Year_Value'] = pollutant_data.loc[
                location_mask, 'Data Value'].shift(12)

            pollutant_data.loc[location_mask, 'Rolling_3M_Avg'] = pollutant_data.loc[
                location_mask, 'Data Value'].rolling(window=3, min_periods=1).mean()
            pollutant_data.loc[location_mask, 'Rolling_12M_Avg'] = pollutant_data.loc[
                location_mask, 'Data Value'].rolling(window=12, min_periods=1).mean()

        le_location = LabelEncoder()
        le_season = LabelEncoder()
        pollutant_data['Location_Encoded'] = le_location.fit_transform(pollutant_data['Geo Place Name'])
        pollutant_data['Season_Encoded'] = le_season.fit_transform(pollutant_data['Season'])

        location_dummies = pd.get_dummies(pollutant_data['Geo Place Name'], prefix='location')
        pollutant_data = pd.concat([pollutant_data, location_dummies], axis=1)

        before_drop = len(pollutant_data)

        pollutant_data = pollutant_data.dropna(subset=['Prev_Month_Value', 'Rolling_3M_Avg'], how='all')

        pollutant_data['Prev_Month_Value'].fillna(pollutant_data['Data Value'], inplace=True)
        pollutant_data['Prev_Year_Value'].fillna(pollutant_data['Data Value'], inplace=True)
        pollutant_data['Rolling_3M_Avg'].fillna(pollutant_data['Data Value'], inplace=True)
        pollutant_data['Rolling_12M_Avg'].fillna(pollutant_data['Data Value'], inplace=True)

        print(f"Dropped {before_drop - len(pollutant_data)} rows with missing values")

        X = pollutant_data[[
            'Year', 'Month', 'Month_sin', 'Month_cos', 'Season_Encoded',
            'Prev_Month_Value', 'Prev_Year_Value', 'Rolling_3M_Avg',
            'Rolling_12M_Avg', 'Location_Encoded'
        ]]

        location_cols = [col for col in pollutant_data.columns if col.startswith('location_')]
        X = pd.concat([X, pollutant_data[location_cols]], axis=1)

        y = pollutant_data['Data Value']

        print(f"Final dataset: {len(X)} samples with {X.shape[1]} features")

        if len(X) == 0:
            raise ValueError("No data available after preprocessing. Check your filtering conditions.")

        return X, y, pollutant_data, le_location, le_season

    except Exception as e:
        print(f"Error in prepare_data_for_prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def train_models(X, y, pollutant_name):

    try:
        # Split data - use a time-based split instead of random for time series data
        train_size = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

        print(f"Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")

        scaler = StandardScaler()
        non_onehot_cols = [col for col in X.columns if not col.startswith('location_')]
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()

        X_train_scaled[non_onehot_cols] = scaler.fit_transform(X_train[non_onehot_cols])
        X_test_scaled[non_onehot_cols] = scaler.transform(X_test[non_onehot_cols])

        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42),
            'XGBoost': XGBRegressor(random_state=42)
        }

        param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100],
                'max_depth': [None, 10]
            },
            'Gradient Boosting': {
                'n_estimators': [50, 100],
                'learning_rate': [0.1],
                'max_depth': [3, 5]
            },
            'XGBoost': {
                'n_estimators': [50, 100],
                'learning_rate': [0.1],
                'max_depth': [3, 5]
            }
        }

        tscv = TimeSeriesSplit(n_splits=3)  # Reduced splits for faster execution

        results = {}
        best_model = None
        best_score = float('inf')

        for name, model in models.items():
            print(f"Training {name}...")

            if name in param_grids:
                grid_search = GridSearchCV(
                    model,
                    param_grids[name],
                    cv=tscv,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1
                )
                grid_search.fit(X_train_scaled, y_train)
                model = grid_search.best_estimator_
                print(f"Best parameters for {name}: {grid_search.best_params_}")
            else:
                model.fit(X_train_scaled, y_train)

            y_pred = model.predict(X_test_scaled)

            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            results[name] = {
                'model': model,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'y_pred': y_pred
            }

            print(f"{name} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}")

            if rmse < best_score:
                best_score = rmse
                best_model = name

        print(f"\nBest model: {best_model} with RMSE of {results[best_model]['rmse']:.2f}")

        return results, X_train_scaled, X_test_scaled, y_train, y_test, best_model, scaler

    except Exception as e:
        print(f"Error in train_models: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def plot_model_comparison(results, pollutant_name, filename):
    """
    Plot a comparison of model performances
    """
    try:
        model_names = list(results.keys())
        rmse_values = [results[name]['rmse'] for name in model_names]
        r2_values = [results[name]['r2'] for name in model_names]

        fig, ax = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
        fig.patch.set_facecolor(COLORS['background'])

        colors = ['#3498DB', '#2ECC71', '#E74C3C', '#9B59B6']
        bars1 = ax[0].bar(model_names, rmse_values, color=colors, alpha=0.8)
        ax[0].set_title('Model Comparison - RMSE\n(lower is better)', fontsize=14, fontweight='bold', pad=15)
        ax[0].set_ylabel('Root Mean Squared Error', fontsize=12)
        ax[0].spines['top'].set_visible(False)
        ax[0].spines['right'].set_visible(False)


        for bar in bars1:
            height = bar.get_height()
            ax[0].text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                       f'{height:.2f}', ha='center', fontsize=10)


        bars2 = ax[1].bar(model_names, r2_values, color=colors, alpha=0.8)
        ax[1].set_title('Model Comparison - R²\n(higher is better)', fontsize=14, fontweight='bold', pad=15)
        ax[1].set_ylabel('R² Score', fontsize=12)
        ax[1].spines['top'].set_visible(False)
        ax[1].spines['right'].set_visible(False)


        for bar in bars2:
            height = bar.get_height()
            ax[1].text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                       f'{height:.2f}', ha='center', fontsize=10)

        plt.suptitle(f'Model Performance Comparison for {pollutant_name}', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()


        output_path = os.path.join(os.getcwd(), filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
        print(f"Saved model comparison plot: {output_path}")
        plt.close()

    except Exception as e:
        print(f"Error in plot_model_comparison: {str(e)}")
        import traceback
        traceback.print_exc()


def plot_predictions(results, best_model, y_test, pollutant_data, pollutant_name, color_key):

    try:
        best_predictions = results[best_model]['y_pred']

        # Create a figure with a good size
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
        fig.patch.set_facecolor(COLORS['background'])
        ax.set_facecolor(COLORS['background'])

        # Set color palette for this pollutant
        cmap = LinearSegmentedColormap.from_list(f"{color_key}_gradient", COLORS[color_key])

        # Plot scatter with style
        scatter = ax.scatter(y_test, best_predictions,
                             alpha=0.7,
                             s=70,
                             c=range(len(y_test)),
                             cmap=cmap,
                             edgecolor='white',
                             linewidth=0.5)

        # Add identity line (perfect predictions)
        min_val = min(y_test.min(), best_predictions.min())
        max_val = max(y_test.max(), best_predictions.max())
        range_val = max_val - min_val
        padding = range_val * 0.05

        line = ax.plot([min_val - padding, max_val + padding],
                       [min_val - padding, max_val + padding],
                       'k--', linewidth=1.5, alpha=0.7)

        # Add confidence intervals (diagonal bands)
        std_dev = np.std(best_predictions - y_test)
        ax.fill_between([min_val - padding, max_val + padding],
                        [min_val - padding - std_dev, max_val + padding - std_dev],
                        [min_val - padding + std_dev, max_val + padding + std_dev],
                        color='gray', alpha=0.1)

        # Style the plot
        pollutant_short = pollutant_name.split('(')[0].strip()
        ax.set_title(f'{pollutant_short} Predictions vs Actual Values\nModel: {best_model}',
                     fontsize=16,
                     fontweight='bold',
                     pad=20)

        ax.set_xlabel('Actual Values', fontsize=14, labelpad=10, fontweight='bold')
        ax.set_ylabel('Predicted Values', fontsize=14, labelpad=10, fontweight='bold')

        # Set limits with padding
        ax.set_xlim(min_val - padding, max_val + padding)
        ax.set_ylim(min_val - padding, max_val + padding)

        # Add metrics annotation
        metrics_text = (
            f"Root Mean Squared Error: {results[best_model]['rmse']:.2f}\n"
            f"Mean Absolute Error: {results[best_model]['mae']:.2f}\n"
            f"R² Score: {results[best_model]['r2']:.2f}"
        )

        ax.text(0.05, 0.95, metrics_text,
                transform=ax.transAxes,
                fontsize=11,
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5',
                          facecolor='white',
                          alpha=0.8,
                          edgecolor=COLORS[color_key][1]))

        # Add over/under prediction annotation
        over_pred = sum(best_predictions > y_test) / len(y_test) * 100
        under_pred = sum(best_predictions < y_test) / len(y_test) * 100

        bias_text = (
            f"Over-predictions: {over_pred:.1f}%\n"
            f"Under-predictions: {under_pred:.1f}%"
        )

        ax.text(0.75, 0.10, bias_text,
                transform=ax.transAxes,
                fontsize=11,
                verticalalignment='bottom',
                horizontalalignment='center',
                bbox=dict(boxstyle='round,pad=0.5',
                          facecolor='white',
                          alpha=0.8,
                          edgecolor='#7F8C8D'))

        # Improve grid appearance
        ax.grid(True, linestyle='--', alpha=0.3)

        # Remove spines
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        # Add a colorbar for time progression
        cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label('Sample Index (Time Progression)', fontsize=10)

        # Save the plot with absolute path
        output_filename = f'model_prediction_{pollutant_name.lower().replace(" ", "_").replace("(", "").replace(")", "")}.png'
        output_path = os.path.join(os.getcwd(), output_filename)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
        print(f"Saved prediction plot: {output_path}")
        plt.close()

    except Exception as e:
        print(f"Error in plot_predictions: {str(e)}")
        import traceback
        traceback.print_exc()


def plot_time_series_prediction(results, best_model, X_test, y_test, pollutant_data, pollutant_name, color_key):
    """
    Plot time series of actual vs predicted values
    """
    try:
        best_predictions = results[best_model]['y_pred']

        # Sort by date for proper time series plotting
        test_df = pd.DataFrame({
            'Actual': y_test.values,
            'Predicted': best_predictions
        })

        # Get the dates corresponding to the test set
        test_indices = y_test.index
        test_dates = pollutant_data.loc[test_indices, 'Start_Date'].reset_index(drop=True)

        test_df['Date'] = test_dates
        test_df = test_df.sort_values('Date')

        # Calculate moving averages for smoother visualization
        window = min(6, len(test_df) // 5)  # Adaptive window size
        window = max(window, 2)  # Ensure window is at least 2
        test_df['Actual_MA'] = test_df['Actual'].rolling(window=window, min_periods=1).mean()
        test_df['Predicted_MA'] = test_df['Predicted'].rolling(window=window, min_periods=1).mean()

        # Create a figure
        fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
        fig.patch.set_facecolor(COLORS['background'])
        ax.set_facecolor(COLORS['background'])

        # Plot actual values
        ax.plot(test_df['Date'], test_df['Actual'], 'o-', color=COLORS[color_key][0],
                alpha=0.4, markersize=4, label='Actual Values')

        # Plot predicted values
        ax.plot(test_df['Date'], test_df['Predicted'], 'o-', color=COLORS[color_key][2],
                alpha=0.4, markersize=4, label='Predicted Values')

        # Plot moving averages for clearer trend lines
        ax.plot(test_df['Date'], test_df['Actual_MA'], '-', color=COLORS[color_key][0],
                linewidth=2.5, label=f'Actual (Moving Avg, n={window})')

        ax.plot(test_df['Date'], test_df['Predicted_MA'], '-', color=COLORS[color_key][2],
                linewidth=2.5, label=f'Predicted (Moving Avg, n={window})')

        # Calculate prediction error
        test_df['Error'] = test_df['Predicted'] - test_df['Actual']

        # Plot error as a light area at the bottom
        ax_twin = ax.twinx()
        ax_twin.fill_between(test_df['Date'], test_df['Error'], 0,
                             where=test_df['Error'] >= 0,
                             alpha=0.2, color='red', label='Over-prediction')

        ax_twin.fill_between(test_df['Date'], test_df['Error'], 0,
                             where=test_df['Error'] < 0,
                             alpha=0.2, color='blue', label='Under-prediction')

        # Set limits for the error axis to be symmetrical
        max_error = max(abs(test_df['Error'].max()), abs(test_df['Error'].min()))
        ax_twin.set_ylim(-max_error * 1.1, max_error * 1.1)
        ax_twin.set_ylabel('Prediction Error', fontsize=12)

        # Add a zero line for the error
        ax_twin.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

        # Format x-axis to show dates nicely
        months = mdates.MonthLocator(interval=3)
        months_fmt = mdates.DateFormatter('%b %Y')
        ax.xaxis.set_major_locator(months)
        ax.xaxis.set_major_formatter(months_fmt)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        # Style the plot
        pollutant_short = pollutant_name.split('(')[0].strip()
        ax.set_title(f'Time Series of {pollutant_short} Predictions\nModel: {best_model}',
                     fontsize=16,
                     fontweight='bold',
                     pad=20)

        ax.set_xlabel('Date', fontsize=14, labelpad=10, fontweight='bold')
        ax.set_ylabel(f'{pollutant_short} Concentration', fontsize=14, labelpad=10, fontweight='bold')

        # Add legend with both axis items
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax_twin.get_legend_handles_labels()
        ax.legend(handles1 + handles2, labels1 + labels2, loc='upper left', frameon=True,
                  facecolor='white', framealpha=0.9, edgecolor='none')

        # Add metrics annotation
        metrics_text = (
            f"RMSE: {results[best_model]['rmse']:.2f}\n"
            f"MAE: {results[best_model]['mae']:.2f}\n"
            f"R²: {results[best_model]['r2']:.2f}"
        )

        ax.text(0.02, 0.97, metrics_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.4',
                          facecolor='white',
                          alpha=0.8,
                          edgecolor=COLORS[color_key][1]))

        # Grid styling
        ax.grid(True, linestyle='--', alpha=0.3)

        # Remove spines
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['top', 'left']:
            ax_twin.spines[spine].set_visible(False)

        # Save the plot with absolute path
        output_filename = f'time_series_prediction_{pollutant_name.lower().replace(" ", "_").replace("(", "").replace(")", "")}.png'
        output_path = os.path.join(os.getcwd(), output_filename)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
        print(f"Saved time series plot: {output_path}")
        plt.close()

    except Exception as e:
        print(f"Error in plot_time_series_prediction: {str(e)}")
        import traceback
        traceback.print_exc()


def plot_feature_importance(results, best_model, X, pollutant_name, color_key):
    """
    Plot feature importance for tree-based models
    """
    try:
        # Check if the best model supports feature importances
        if best_model not in ['Random Forest', 'Gradient Boosting', 'XGBoost']:
            print(f"Feature importance not available for {best_model}")
            return

        # Get the model and its feature importances
        model = results[best_model]['model']
        importances = model.feature_importances_

        # Get feature names
        feature_names = X.columns

        # Create a dataframe for visualization
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)

        # Filter out low importance features for clarity
        top_features = importance_df.head(15)

        # Create color gradient based on importance
        cmap = LinearSegmentedColormap.from_list(f"{color_key}_gradient", COLORS[color_key])
        colors = cmap(np.linspace(0.2, 0.8, len(top_features)))

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
        fig.patch.set_facecolor(COLORS['background'])
        ax.set_facecolor(COLORS['background'])

        # Plot horizontal bars
        bars = ax.barh(top_features['Feature'], top_features['Importance'],
                       color=colors, alpha=0.8, height=0.6)

        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width * 1.01, bar.get_y() + bar.get_height() / 2,
                    f'{width:.3f}', va='center', fontsize=10)

        # Style the plot
        pollutant_short = pollutant_name.split('(')[0].strip()
        ax.set_title(f'Feature Importance for {pollutant_short} Prediction\nModel: {best_model}',
                     fontsize=16, fontweight='bold', pad=20)

        ax.set_xlabel('Importance', fontsize=14, labelpad=10, fontweight='bold')
        ax.set_ylabel('Feature', fontsize=14, labelpad=10, fontweight='bold')

        # Improve grid appearance
        ax.grid(True, linestyle='--', alpha=0.3)

        # Remove spines
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        # Save the plot with absolute path
        output_filename = f'feature_importance_{pollutant_name.lower().replace(" ", "_").replace("(", "").replace(")", "")}.png'
        output_path = os.path.join(os.getcwd(), output_filename)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
        print(f"Saved feature importance plot: {output_path}")
        plt.close()

    except Exception as e:
        print(f"Error in plot_feature_importance: {str(e)}")
        import traceback
        traceback.print_exc()


def plot_longterm_trend(future_df, pollutant_data, pollutant_name, color_key):
    """
    Plot the long-term trend of predictions to show where we're heading
    """
    try:
        fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
        fig.patch.set_facecolor(COLORS['background'])
        ax.set_facecolor(COLORS['background'])

        # Calculate the min/max years in the dataset
        min_year = pollutant_data['Start_Date'].dt.year.min()
        max_year = future_df['Date'].dt.year.max()

        # Create a gradient background showing time progression
        gradient = np.linspace(0, 1, 500).reshape(-1, 1)
        extent = [min_year - 0.5, max_year + 0.5, 0, 15]  # Adjust y-range as needed
        ax.imshow(gradient, aspect='auto', extent=extent,
                  alpha=0.1, cmap=LinearSegmentedColormap.from_list("time_gradient",
                                                                    ['#F8F9F9', '#ECF0F1']))

        # Prepare historical data - yearly averages across all locations
        historical_yearly = pollutant_data.groupby(pollutant_data['Start_Date'].dt.year)['Data Value'].mean()
        historical_years = historical_yearly.index.tolist()
        historical_values = historical_yearly.values

        # Prepare future data - yearly averages across all locations
        # Group by year and calculate the mean
        future_df['Year'] = future_df['Date'].dt.year
        future_yearly = future_df.groupby('Year')['Predicted_Value'].mean()
        future_years = future_yearly.index.tolist()
        future_values = future_yearly.values

        # Get combined years for trend line
        all_years = historical_years + future_years

        # Create color maps for past and future
        past_cmap = LinearSegmentedColormap.from_list("past", ['#3498DB', '#2E86C1', '#2874A6'])
        future_cmap = LinearSegmentedColormap.from_list("future", COLORS[color_key])

        # Plot historical data with a different color/style
        ax.plot(historical_years, historical_values, 'o-',
                color=past_cmap(0.7),
                linewidth=3,
                label='Historical Data',
                markersize=8)

        # Add shaded area for historical uncertainty
        historical_std = pollutant_data.groupby(pollutant_data['Start_Date'].dt.year)['Data Value'].std().fillna(0)
        ax.fill_between(historical_years,
                        historical_values - historical_std,
                        historical_values + historical_std,
                        color=past_cmap(0.7),
                        alpha=0.2)

        # Plot prediction line with a gradient effect to show uncertainty increasing with time
        points = np.array([future_years, future_values]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create a LineCollection with a gradient color
        from matplotlib.collections import LineCollection
        norm = plt.Normalize(0, len(future_years))
        lc = LineCollection(segments, cmap=future_cmap, linewidth=3)
        lc.set_array(np.arange(len(future_years)))
        line = ax.add_collection(lc)

        # Add future data points
        sc = ax.scatter(future_years, future_values,
                        c=np.arange(len(future_years)),
                        cmap=future_cmap,
                        s=80,
                        label='Predicted Future',
                        zorder=5,
                        edgecolor='white',
                        linewidth=1)

        # Add uncertainty cone that widens with time
        # Calculate the standard deviation among locations for each future year
        future_yearly_std = future_df.groupby('Year')['Predicted_Value'].std().fillna(0)

        # Add extra uncertainty that grows with time
        time_uncertainty = np.linspace(0, future_yearly_std.mean() * 3, len(future_years))
        total_uncertainty = future_yearly_std + time_uncertainty

        ax.fill_between(future_years,
                        future_values - total_uncertainty,
                        future_values + total_uncertainty,
                        color=future_cmap(0.5),
                        alpha=0.2,
                        label='Prediction Uncertainty')

        # Fit a trend line to the entire dataset (past + future)
        all_values = np.concatenate([historical_values, future_values])
        z = np.polyfit(all_years, all_values, 1)
        p = np.poly1d(z)
        trend_values = p(all_years)

        # Plot the trend line as a dashed line
        ax.plot(all_years, trend_values, '--',
                color='#7F8C8D',
                linewidth=2,
                label='Overall Trend',
                alpha=0.8)

        # Add vertical line to separate past from future
        current_year = pollutant_data['Start_Date'].dt.year.max()
        ax.axvline(x=current_year, color='#7F8C8D', linestyle='-', linewidth=1.5, alpha=0.5)
        ax.text(current_year + 0.1, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02,
                'Present', fontsize=10, color='#7F8C8D')

        # Add shaded regions showing EPA standards or safe levels if available
        if pollutant_name == 'Fine particles (PM 2.5)':
            # EPA annual standard for PM2.5 is 12 µg/m³
            ax.axhspan(0, 12, alpha=0.1, color='#2ECC71', label='EPA Standard (Safe)')
            ax.axhspan(12, 35, alpha=0.1, color='#F39C12', label='Moderate')
            ax.axhspan(35, ax.get_ylim()[1], alpha=0.1, color='#E74C3C', label='Unhealthy')
        elif pollutant_name == 'Ozone (O3)':
            # EPA 8-hour standard for Ozone is 0.070 ppm (70 ppb)
            # Since your data might be in different units, adjust accordingly
            ax.axhspan(0, 25, alpha=0.1, color='#2ECC71', label='Good')
            ax.axhspan(25, 50, alpha=0.1, color='#F39C12', label='Moderate')
            ax.axhspan(50, ax.get_ylim()[1], alpha=0.1, color='#E74C3C', label='Unhealthy')
        elif pollutant_name == 'Nitrogen dioxide (NO2)':
            # EPA 1-hour standard for NO2 is 100 ppb
            ax.axhspan(0, 10, alpha=0.1, color='#2ECC71', label='Good')
            ax.axhspan(10, 20, alpha=0.1, color='#F39C12', label='Moderate')
            ax.axhspan(20, ax.get_ylim()[1], alpha=0.1, color='#E74C3C', label='Unhealthy')

        # Calculate the annual percentage change
        years_diff = all_years[-1] - all_years[0]
        value_change = p(all_years[-1]) - p(all_years[0])
        annual_percent_change = (value_change / p(all_years[0])) / years_diff * 100

        # Add an annotation about the trend
        if annual_percent_change < 0:
            trend_text = (f"Long-term trend: IMPROVING\n"
                          f"Annual reduction rate: {abs(annual_percent_change):.2f}% per year\n"
                          f"Projected total change by {all_years[-1]}: {value_change:.2f} units")
            trend_color = '#27AE60'  # Green for improvement
        else:
            trend_text = (f"Long-term trend: WORSENING\n"
                          f"Annual increase rate: {annual_percent_change:.2f}% per year\n"
                          f"Projected total change by {all_years[-1]}: +{value_change:.2f} units")
            trend_color = '#C0392B'  # Red for worsening

        # Create a text box with the trend information
        props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=trend_color)
        ax.text(0.02, 0.96, trend_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)

        # Style the plot
        pollutant_short = pollutant_name.split('(')[0].strip()
        ax.set_title(f'Long-Term {pollutant_short} Trend Projection ({min_year}-{max_year})',
                     fontsize=18, fontweight='bold', pad=20)

        ax.set_xlabel('Year', fontsize=14, labelpad=10, fontweight='bold')
        ax.set_ylabel(f'{pollutant_short} Concentration', fontsize=14, labelpad=10, fontweight='bold')

        # Set x-axis to show years
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        # Add legend
        ax.legend(loc='best', frameon=True, fontsize=10)

        # Add annotation for increasing uncertainty
        ax.text(0.98, 0.05, 'Uncertainty increases\nwith prediction time',
                transform=ax.transAxes, fontsize=10, ha='right',
                style='italic', color='#7F8C8D',
                bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.5', edgecolor='none'))

        # Improve grid styling
        ax.grid(True, linestyle='--', alpha=0.3)

        # Remove spines
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        # Add an arrow pointing to the future to emphasize the direction
        future_midpoint = (future_years[0] + future_years[-1]) / 2
        future_value = p(future_midpoint)
        ax.annotate('Future Trajectory',
                    xy=(future_years[-1], future_values[-1]),
                    xytext=(future_midpoint, future_value + 2),
                    arrowprops=dict(facecolor=future_cmap(0.8), shrink=0.05, width=2, headwidth=8),
                    ha='center', fontsize=10, fontweight='bold')


        # Save the plot
        output_filename = f'longterm_projection_{pollutant_name.lower().replace(" ", "_").replace("(", "").replace(")", "")}.png'
        output_path = os.path.join(os.getcwd(), output_filename)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
        print(f"Saved long-term trend plot: {output_path}")
        plt.close()

    except Exception as e:
        print(f"Error in plot_longterm_trend: {str(e)}")
        import traceback
        traceback.print_exc()

def predict_future(results, best_model, pollutant_data, le_location, le_season, scaler, color_key, pollutant_name):
    """
    Make predictions for future periods (up to 10 years)
    """
    try:
        # Get the best model
        model = results[best_model]['model']

        # Get latest date in the data
        last_date = pollutant_data['Start_Date'].max()
        print(f"Last date in data: {last_date}")

        # Create a date range for future predictions (10 years monthly)
        prediction_years = 10
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1),
                                    periods=12 * prediction_years,
                                    freq='ME')
        print(f"Predicting for {len(future_dates)} future dates from {future_dates[0]} to {future_dates[-1]}")

        # Get locations from data (preferring main boroughs if available)
        all_locations = pollutant_data['Geo Place Name'].unique()
        main_boroughs = ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island']
        locations_to_predict = [loc for loc in all_locations if loc in main_boroughs]

        if not locations_to_predict:
            # If no main boroughs, use top locations with most data points
            location_counts = pollutant_data['Geo Place Name'].value_counts()
            locations_to_predict = location_counts.index[:5].tolist()

        print(f"Making predictions for locations: {locations_to_predict}")

        # Store the column names used during model training to ensure same order
        feature_cols = []

        # Get non-one-hot encoded columns
        base_cols = [
            'Year', 'Month', 'Month_sin', 'Month_cos', 'Season_Encoded',
            'Prev_Month_Value', 'Prev_Year_Value', 'Rolling_3M_Avg',
            'Rolling_12M_Avg', 'Location_Encoded'
        ]
        feature_cols.extend(base_cols)

        # Get location dummy columns that were in the training data
        location_cols = [col for col in pollutant_data.columns if col.startswith('location_')]
        feature_cols.extend(location_cols)

        # Get all seasons present in training data
        known_seasons = set(pollutant_data['Season'].unique())
        print(f"Seasons in training data: {known_seasons}")

        # Default season encoding if a season is not in training data
        default_season_encoding = 0
        if len(le_season.classes_) > 0:
            default_season_encoding = le_season.transform([le_season.classes_[0]])[0]

        # Create dataframe for future predictions
        future_rows = []

        # Loop through each location
        for location in locations_to_predict:
            # Get location-specific data
            loc_data = pollutant_data[pollutant_data['Geo Place Name'] == location].sort_values('Start_Date')

            if len(loc_data) < 12:
                print(f"Skipping location {location} due to insufficient history")
                continue  # Skip locations with insufficient history

            # Get the most recent values for this location to use as initial lag features
            latest_values = loc_data.tail(12)['Data Value'].values
            latest_date = loc_data['Start_Date'].max()

            # Initial lag values (will be updated as we predict)
            prev_month_value = loc_data.iloc[-1]['Data Value']
            prev_year_value = loc_data[loc_data['Start_Date'] <= latest_date - pd.DateOffset(months=12)].iloc[-1][
                'Data Value'] if len(loc_data) >= 12 else prev_month_value
            rolling_3m = loc_data.tail(3)['Data Value'].mean()
            rolling_12m = loc_data.tail(12)['Data Value'].mean() if len(loc_data) >= 12 else rolling_3m

            # Get encoded values
            location_encoded = le_location.transform([location])[0]

            # For each future date
            for i, date in enumerate(future_dates):
                # Extract date features
                year = date.year
                month = date.month
                month_sin = np.sin(2 * np.pi * month / 12)
                month_cos = np.cos(2 * np.pi * month / 12)

                # Determine season
                season = 'Winter' if month in [12, 1, 2] else 'Spring' if month in [3, 4, 5] else 'Summer' if month in [
                    6, 7, 8] else 'Fall'

                # Check if this season exists in the training data
                if season in known_seasons:
                    try:
                        season_encoded = le_season.transform([season])[0]
                    except (ValueError, KeyError):
                        # If there's an error, use the default encoding
                        print(f"Warning: Season '{season}' not found in encoder. Using default value.")
                        season_encoded = default_season_encoding
                else:
                    # Season not in training data, use default
                    print(f"Warning: Season '{season}' not in training data. Using default value.")
                    season_encoded = default_season_encoding

                # Create feature row (same format as training data)
                feature_row = {
                    'Year': year,
                    'Month': month,
                    'Month_sin': month_sin,
                    'Month_cos': month_cos,
                    'Season_Encoded': season_encoded,
                    'Prev_Month_Value': prev_month_value,
                    'Prev_Year_Value': prev_year_value,
                    'Rolling_3M_Avg': rolling_3m,
                    'Rolling_12M_Avg': rolling_12m,
                    'Location_Encoded': location_encoded,
                    'Location': location,
                    'Date': date
                }

                # Add one-hot encoding for location (initialize all to 0)
                for loc in all_locations:
                    feature_row[f'location_{loc}'] = 0

                # Set the correct location to 1
                feature_row[f'location_{location}'] = 1

                future_rows.append(feature_row)

        # Create dataframe for future data
        future_df = pd.DataFrame(future_rows)

        # Make a copy of the DataFrame to avoid fragmentation warnings
        future_df_copy = future_df.copy()

        # Get features in the same format and order as training data
        X_future = pd.DataFrame()

        # First add the base feature columns in the same order
        X_future = future_df_copy[base_cols].copy()

        # Then add location dummy columns in the same order as they appeared in training
        location_data = {}
        for col in location_cols:
            if col in future_df_copy.columns:
                location_data[col] = future_df_copy[col].values
            else:
                # If a location column from training is missing in future data, add it with zeros
                location_data[col] = np.zeros(len(future_df_copy))

        # Add all location columns at once to avoid fragmentation
        location_df = pd.DataFrame(location_data, index=X_future.index)
        X_future = pd.concat([X_future, location_df], axis=1)

        # Fill any NaN values (required for most sklearn models)
        X_future = X_future.ffill()  # Forward-fill
        X_future = X_future.bfill()  # Backward-fill
        X_future.fillna(0, inplace=True)  # Fill any remaining NaNs with 0

        # Scale the data using the same scaler as training data
        X_future_scaled = X_future.copy()
        X_future_scaled[base_cols] = scaler.transform(X_future[base_cols])

        # Make predictions
        future_predictions = model.predict(X_future_scaled)
        future_df['Predicted_Value'] = future_predictions

        # Plot future predictions (both detailed and long-term)
        plot_future_predictions(future_df, pollutant_name, color_key)
        plot_longterm_trend(future_df, pollutant_data, pollutant_name, color_key)

        # Skipping CSV creation to avoid file clutter
        print(f"Future predictions calculated for {pollutant_name}")

        return future_df

    except Exception as e:
        print(f"Error in predict_future: {str(e)}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()  # Return empty dataframe on error

def plot_future_predictions(future_df, pollutant_name, color_key):
    """
    Plot future predictions for each main location
    """
    try:
        fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
        fig.patch.set_facecolor(COLORS['background'])
        ax.set_facecolor(COLORS['background'])

        # Get the main locations
        main_locations = future_df['Location'].unique()

        # Create colormap
        cmap = LinearSegmentedColormap.from_list(f"{color_key}_gradient", COLORS[color_key])
        colors = [cmap(i / len(main_locations)) for i in range(len(main_locations))]

        # Plot each location
        for i, location in enumerate(main_locations):
            loc_data = future_df[future_df['Location'] == location].sort_values('Date')

            # Calculate moving average for smoother line
            loc_data['Pred_MA'] = loc_data['Predicted_Value'].rolling(window=3, min_periods=1).mean()

            # Plot the data
            ax.plot(loc_data['Date'], loc_data['Pred_MA'], '-',
                    color=colors[i],
                    linewidth=2.5,
                    label=location)

            # Add data points
            ax.scatter(loc_data['Date'], loc_data['Predicted_Value'],
                       color=colors[i],
                       alpha=0.5,
                       s=30)

        # Style the plot
        pollutant_short = pollutant_name.split('(')[0].strip()
        ax.set_title(f'Future {pollutant_short} Predictions by Location',
                     fontsize=16,
                     fontweight='bold',
                     pad=20)

        ax.set_xlabel('Date', fontsize=14, labelpad=10, fontweight='bold')
        ax.set_ylabel(f'Predicted {pollutant_short} Concentration', fontsize=14, labelpad=10, fontweight='bold')

        # Format x-axis to show dates nicely
        months = mdates.MonthLocator(interval=3)
        months_fmt = mdates.DateFormatter('%b %Y')
        ax.xaxis.set_major_locator(months)
        ax.xaxis.set_major_formatter(months_fmt)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        # Add legend
        ax.legend(loc='best', frameon=True, facecolor='white', framealpha=0.9, edgecolor='none')

        # Add annotation for uncertainty
        ax.text(0.02, 0.02,
                "Note: Predictions become less certain further into the future.",
                transform=ax.transAxes,
                fontsize=10,
                style='italic',
                bbox=dict(boxstyle='round,pad=0.4',
                          facecolor='white',
                          alpha=0.8,
                          edgecolor='#7F8C8D'))

        # Grid styling
        ax.grid(True, linestyle='--', alpha=0.3)

        # Remove spines
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        # Save the plot
        output_filename = f'future_predictions_{pollutant_name.lower().replace(" ", "_").replace("(", "").replace(")", "")}.png'
        output_path = os.path.join(os.getcwd(), output_filename)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
        print(f"Saved future predictions plot: {output_path}")
        plt.close()

    except Exception as e:
        print(f"Error in plot_future_predictions: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    """
    Main function to run the enhanced air quality prediction
    """
    try:
        # Load data
        print("\n" + "=" * 70)
        print("ENHANCED AIR QUALITY PREDICTION MODEL")
        print("=" * 70)

        print("\nLoading data...")
        df = pd.read_csv('Air_Quality.csv')
        print(f"Data loaded successfully: {len(df)} rows")

        # Convert dates
        df['Start_Date'] = pd.to_datetime(df['Start_Date'])

        # List of pollutants to analyze with their color keys
        pollutants = [
            ('Fine particles (PM 2.5)', 'pm25'),
            ('Ozone (O3)', 'ozone'),
            ('Nitrogen dioxide (NO2)', 'no2')
        ]

        for pollutant_name, color_key in pollutants:
            print(f"\n{'=' * 50}")
            print(f"ANALYZING {pollutant_name}")
            print(f"{'=' * 50}")

            # 1. Prepare data with enhanced features
            print("\nPreparing data with enhanced features...")
            X, y, pollutant_data, le_location, le_season = prepare_data_for_prediction(df, pollutant_name)

            # 2. Train and compare multiple models
            print("\nTraining and comparing multiple models...")
            results, X_train, X_test, y_train, y_test, best_model, scaler = train_models(X, y, pollutant_name)

            # 3. Plot model comparison
            print("\nCreating model comparison visualization...")
            comparison_filename = f'model_comparison_{pollutant_name.lower().replace(" ", "_").replace("(", "").replace(")", "")}.png'
            plot_model_comparison(results, pollutant_name, comparison_filename)

            # 4. Plot predictions for best model
            print("\nCreating prediction visualizations...")
            plot_predictions(results, best_model, y_test, pollutant_data, pollutant_name, color_key)

            # 5. Plot time series predictions
            print("Creating time series visualization...")
            plot_time_series_prediction(results, best_model, X_test, y_test, pollutant_data, pollutant_name, color_key)

            # 6. Plot feature importance (if applicable)
            print("Creating feature importance visualization...")
            plot_feature_importance(results, best_model, X, pollutant_name, color_key)

            # 7. Make future predictions
            print("\nMaking future predictions...")
            future_df = predict_future(results, best_model, pollutant_data, le_location, le_season, scaler, color_key,
                                       pollutant_name)

        # Replace the print section at the end of main() with:

        print("\nAll analyses completed successfully!")
        print("\nGenerated Files:")
        for pollutant_name, _ in pollutants:
            clean_name = pollutant_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
            print(f"  - model_comparison_{clean_name}.png")
            print(f"  - model_prediction_{clean_name}.png")
            print(f"  - time_series_prediction_{clean_name}.png")
            if best_model != "Linear Regression":  # Only show if feature importance was generated
                print(f"  - feature_importance_{clean_name}.png")
            print(f"  - future_predictions_{clean_name}.png")
            print(f"  - longterm_projection_{clean_name}.png")  # Added this line
            # Removed CSV file from the list

    except Exception as e:
        print(f"\nERROR in main program: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()