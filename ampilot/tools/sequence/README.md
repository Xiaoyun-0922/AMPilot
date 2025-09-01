agent/
  tools/
    sequence/
      README.md
      ingest_normalize/      # 摄取与规范化
      homology_search/       # 同源检索（多后端适配）
      clustering_derep/      # 聚类与去冗余
      msa/                   # 多序列比对（MSA）
      trimming_masking/      # 对齐裁剪与掩码
      profiles/              # PSSM / HMM
      conservation_coev/     # 保守性 & 协同演化
      motifs_domains/        # motif / 区段标注（TM/IDR/Signal）
      featurization/         # K-mer / AAindex / LM embedding
      consensus_variants/    # 共识序列 & 变体约束
      phylogeny_optional/    # 系统发育（可选）
      commons/               # 契约(schema)、指标(metrics)、缓存、预设(presets)
