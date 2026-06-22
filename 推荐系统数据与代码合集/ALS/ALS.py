import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


# 1. 数据读取与基本统计
ratings = pd.read_csv(
    "./ml-100k/u.data",
    sep="\t",
    names=["user_id", "item_id", "rating", "timestamp"],
)
ratings_raw = ratings.copy()
print(ratings.head())

# 电影元数据（用于结果展示）
movies = pd.read_csv("./ml-100k/u.item", sep="|", encoding="latin-1", header=None)
movie_columns = [
    "movie_id",
    "title",
    "release_date",
    "video_release_date",
    "imdb_url",
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]
genre_columns = movie_columns[5:]
movies.columns = movie_columns
movies["movie_id"] = movies["movie_id"].astype(int)
movies[genre_columns] = movies[genre_columns].astype(int)
movies["genres"] = movies[genre_columns].apply(
    lambda row: "|".join([genre for genre, flag in zip(genre_columns, row) if flag == 1])
    or "Unknown",
    axis=1,
)
movies_metadata = movies[["movie_id", "title", "genres"]]

# 2. 用户与物品编码
user_encoder = LabelEncoder()
item_encoder = LabelEncoder()
ratings["user_id"] = user_encoder.fit_transform(ratings["user_id"])
ratings["item_id"] = item_encoder.fit_transform(ratings["item_id"])
num_users = ratings["user_id"].nunique()
num_items = ratings["item_id"].nunique()

# 3. 构建训练 / 测试集矩阵
train_df, test_df = train_test_split(
    ratings[["user_id", "item_id", "rating"]], test_size=0.2, random_state=42
)


def build_matrix(df, num_users, num_items):
    mat = np.zeros((num_users, num_items), dtype=np.float32)
    for row in df.itertuples(index=False):
        mat[row.user_id, row.item_id] = row.rating
    return mat


train_matrix = build_matrix(train_df, num_users, num_items)
test_matrix = build_matrix(test_df, num_users, num_items)
print(f"用户数量: {num_users}, 物品数量: {num_items}")


# 4. 交替最小二乘训练
def run_als(train_matrix, test_matrix, k=20, reg=0.1, epochs=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    R_train = torch.tensor(train_matrix, dtype=torch.float32, device=device)
    R_test = torch.tensor(test_matrix, dtype=torch.float32, device=device)
    train_mask = (R_train > 0).float()
    test_mask = (R_test > 0).float()
    num_users, num_items = R_train.shape

    torch.manual_seed(42)
    U = torch.randn(num_users, k, device=device) * 0.01
    V = torch.randn(num_items, k, device=device) * 0.01
    I_k = torch.eye(k, device=device)

    train_history, test_history = [], []

    for epoch in range(epochs):
        # 更新用户隐向量
        for u in range(num_users):
            item_idx = torch.nonzero(train_mask[u], as_tuple=False).squeeze()
            if item_idx.numel() == 0:
                continue
            if item_idx.ndim == 0:
                item_idx = item_idx.unsqueeze(0)
            V_u = V[item_idx]
            r_u = R_train[u, item_idx]
            A = V_u.T @ V_u + reg * I_k
            b = V_u.T @ r_u
            U[u] = torch.linalg.solve(A, b)

        # 更新物品隐向量
        for i in range(num_items):
            user_idx = torch.nonzero(train_mask[:, i], as_tuple=False).squeeze()
            if user_idx.numel() == 0:
                continue
            if user_idx.ndim == 0:
                user_idx = user_idx.unsqueeze(0)
            U_i = U[user_idx]
            r_i = R_train[user_idx, i]
            A = U_i.T @ U_i + reg * I_k
            b = U_i.T @ r_i
            V[i] = torch.linalg.solve(A, b)

        preds = U @ V.T
        train_rmse = torch.sqrt(
            ((train_mask * (R_train - preds)) ** 2).sum() / train_mask.sum()
        )
        if test_mask.sum() > 0:
            test_rmse = torch.sqrt(
                ((test_mask * (R_test - preds)) ** 2).sum() / test_mask.sum()
            )
        else:
            test_rmse = torch.tensor(0.0, device=device)

        train_history.append(train_rmse.item())
        test_history.append(test_rmse.item())
        print(
            f"Epoch {epoch + 1:02d}/{epochs} - Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}"
        )

    return U.cpu(), V.cpu(), train_history, test_history


U, V, train_rmse_history, test_rmse_history = run_als(
    train_matrix, test_matrix, k=20, reg=0.1, epochs=30
)


# 5. 损失（RMSE）曲线
def plot_rmse_curve(train_history, test_history, output_path="./ALS/als_rmse_curve.png"):
    plt.figure(figsize=(10, 5))
    epochs = range(1, len(train_history) + 1)
    plt.plot(epochs, train_history, label="Train RMSE", marker="o")
    plt.plot(epochs, test_history, label="Test RMSE", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title("ALS Training vs Test RMSE")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.savefig(output_path, dpi=200)
    plt.show()


plot_rmse_curve(train_rmse_history, test_rmse_history)


# 6. 推荐函数
pred_matrix = U.numpy() @ V.numpy().T


def recommend_top_n(user_idx, pred_scores, train_matrix, top_n=10):
    scores = pred_scores[user_idx].copy()
    seen_mask = train_matrix[user_idx] > 0
    scores[seen_mask] = -np.inf
    top_items = np.argsort(scores)[::-1][:top_n]
    return top_items


# 7. 用户历史与推荐结果展示
encoded_user_idx = 1
original_user_id = user_encoder.inverse_transform([encoded_user_idx])[0]

user_history = (
    ratings_raw[ratings_raw["user_id"] == original_user_id]
    .merge(movies_metadata, left_on="item_id", right_on="movie_id", how="left")
    [["item_id", "rating", "title", "genres"]]
    .sort_values(by="rating", ascending=False)
)

print("用户0历史观影记录（按评分从高到低）：")
print(user_history.to_string(index=False))

recommended_item_indices = recommend_top_n(
    encoded_user_idx, pred_matrix, train_matrix, top_n=10
)
original_item_ids = item_encoder.inverse_transform(recommended_item_indices)

recommendations_df = pd.DataFrame({"item_id": original_item_ids})
recommendations_df = recommendations_df.merge(
    movies_metadata, left_on="item_id", right_on="movie_id", how="left"
)[["item_id", "title", "genres"]]

print("为用户0推荐的前10个电影：")
print(recommendations_df.to_string(index=False))

