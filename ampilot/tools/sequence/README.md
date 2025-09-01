## 模块结构

```
sequence/
  ingest_normalize/       # 序列摄取与清洗
  homology_search/        # 同源检索
  clustering_derep/       # 聚类 & 去冗余
  msa/                    # 多序列比对
  trimming_masking/       # 对齐裁剪 & 掩码
  profiles/               # PSSM / HMM
  conservation_coev/      # 保守性 & 共进化
  motifs_domains/         # Motif & 功能域
  featurization/          # 特征化 (k-mer, embedding)
  consensus_variants/     # 共识序列 & 变体约束
  phylogeny_optional/     # 系统发育 (可选)
```

---

## 开发 Checklist

每个子模块应实现 **一个主函数**，输入/输出必须符合统一的 **schema（定义在 commons/），並且目前都統一使用MCP 格式來進行包裝**。

### 1. ingest\_normalize

* **作用**：载入序列，清洗 & 质控。
* **方法**：去重复、检查合法性、低复杂度标记。
* **输入**：FASTA/序列列表。
* **输出**：`SequenceBatch` + `QCReport`。

### 2. homology\_search

* **作用**：查找同源序列。
* **方法**：MMseqs2 / BLAST+ / HMMER。
* **输入**：单序列或代表序列。
* **输出**：`HomologSet`。

### 3. clustering\_derep

* **作用**：聚类并挑代表，去除冗余。
* **方法**：CD-HIT / MMseqs2 cluster。
* **输入**：`HomologSet`。
* **输出**：`ClusterSet`、`DedupBatch`。

### 4. msa

* **作用**：对齐序列。
* **方法**：MAFFT / Clustal Omega / HHblits。
* **输入**：`DedupBatch`。
* **输出**：`Alignment`、`MSAMetrics`。

### 5. trimming\_masking

* **作用**：裁剪对齐，去掉低质量列。
* **方法**：gap% 阈值 / 熵过滤。
* **输入**：`Alignment`。
* **输出**：`TrimmedAlignment`、`Mask`。

### 6. profiles

* **作用**：生成 profiles。
* **方法**：PSSM (PSI-BLAST) / HMM (HMMER3)。
* **输入**：`Alignment`。
* **输出**：`PSSM`、`HMMProfile`。

### 7. conservation\_coev

* **作用**：分析保守性 & 共进化。
* **方法**：Shannon entropy / MI / DI / PLM。
* **输入**：`Alignment`。
* **输出**：`ConservationScores`、`CouplingMatrix`。

### 8. motifs\_domains

* **作用**：识别 motif & 功能域。
* **方法**：PROSITE / Pfam / TMHMM / IUPred / SignalP。
* **输入**：`Alignment` 或单序列。
* **输出**：`MotifSet`、`RegionAnnotations`。

### 9. featurization

* **作用**：序列特征化。
* **方法**：k-mer、AAindex、LM embedding (ESM, ProtBERT)。
* **输入**：`SequenceBatch` 或 `Alignment`。
* **输出**：`FeatureTable`、`EmbeddingMatrix`。

### 10. consensus\_variants

* **作用**：生成共识序列 & 变体约束。
* **方法**：基于频率矩阵 / profile。
* **输入**：`Alignment`。
* **输出**：`ConsensusSequence`、`VariantConstraints`。

### 11. phylogeny\_optional

* **作用**：构建粗略系统发育树（解释性用途）。
* **方法**：FastTree / Neighbor-Joining。
* **输入**：`Alignment`。
* **输出**：`PhyloTree`、`DistanceMatrix`。
