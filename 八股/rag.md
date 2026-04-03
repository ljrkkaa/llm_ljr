这里涉及 Milvus Schema 和 索引 (Index) 的区别，以及你代码里 build_index 做的事情，我帮你梳理一下：

1️⃣ Schema 的作用

Schema 定义的是 数据结构和存储字段信息，包括：

字段名（field_name）：比如 text_dense、summary_dense、text_sparse。

数据类型（DataType.FLOAT_VECTOR、DataType.VARCHAR 等）。

主键和是否自增（auto_id、is_primary）。

是否启用动态字段（enable_dynamic_field）。

可以加 函数 (Function)，比如 BM25 或其他向量化函数，把原始数据转换成向量。

换句话说，Schema 决定了你在 Milvus 里能存储什么样的数据。

2️⃣ Index 的作用

索引是为了 加速搜索，尤其是向量检索或稀疏向量检索：

Schema 定义了字段，但如果没有索引，Milvus 会做 全表扫描，速度很慢。

索引根据向量类型选择算法，比如：

HNSW：用于稠密向量近似最近邻搜索 (Approximate Nearest Neighbor, ANN)。

SPARSE_INVERTED_INDEX：用于稀疏向量或 BM25 索引，加速关键字匹配。

指定参数控制索引效果：

M, efConstruction（HNSW 的参数，影响精度和构建时间）。

inverted_index_algo, bm25_k1, bm25_b（稀疏向量/BM25 的参数）。

创建索引后，再 load_collection，才能用 Milvus 检索时快速返回结果。