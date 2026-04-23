import subprocess, sys

for pkg in ["numpy", "matplotlib", "seaborn"]:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

# ── Imports ─────
import time, io, os
import boto3
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, LongType, DoubleType
)
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
    ClusteringEvaluator,
)
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline

# CONFIGURATION
S3_BUCKET   = "my-flame-emr-bucket"
S3_PREFIX   = "cryptocurrency team/trial2"
S3_BASE     = f"s3://{S3_BUCKET}/{S3_PREFIX}"
OUTPUT_BASE = f"{S3_BASE}/output"

TXS_CLASSES_PATH     = f"{S3_BASE}/txs_classes.csv"
TXS_EDGELIST_PATH    = f"{S3_BASE}/txs_edgelist.csv"
TXS_FEATURES_PATH    = f"{S3_BASE}/txs_features.csv"
WALLETS_CLASSES_PATH = f"{S3_BASE}/wallets_classes.csv"
ADDRADDR_PATH        = f"{S3_BASE}/AddrAddr_edgelist.csv"
ADDRTX_PATH          = f"{S3_BASE}/AddrTx_edgelist.csv"
TXADDR_PATH          = f"{S3_BASE}/TxAddr_edgelist.csv"

ILLICIT      = 1
LICIT        = 2
UNKNOWN      = 3
TRAIN_CUTOFF = 34
SEP          = "=" * 72

C_ILLICIT = "#E53935"
C_LICIT   = "#43A047"
C_UNKNOWN = "#FFA726"
C_BLUE    = "#1565C0"
C_PURPLE  = "#6A1B9A"

sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 11, "figure.dpi": 150})


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def save_fig(fig, filename):
    key = f"{S3_PREFIX}/output/charts/{filename}"
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    boto3.client("s3").put_object(Bucket=S3_BUCKET, Key=key, Body=buf.read())
    print(f"  Chart → s3://{S3_BUCKET}/{key}")
    plt.close(fig)


def save_csv(data, filename):
    import pandas as pd
    if isinstance(data, dict):
        data = pd.DataFrame([data])
    key = f"{S3_PREFIX}/output/csv/{filename}"
    buf = io.StringIO()
    data.to_csv(buf, index=False)
    boto3.client("s3").put_object(
        Bucket=S3_BUCKET, Key=key, Body=buf.getvalue().encode("utf-8")
    )
    print(f"  CSV  → s3://{S3_BUCKET}/{key}  ({len(data)} rows)")
    return data


def spark_to_csv(df, filename):
    pdf = df.toPandas()
    save_csv(pdf, filename)
    return pdf


# ══════════════════════════════════════════════════════════════════════════════
# 0. SPARK SESSION
# ══════════════════════════════════════════════════════════════════════════════
def create_spark_session():
    spark = SparkSession.builder \
        .appName("Elliptic_BigData_FraudDetection") \
        .config("spark.sql.shuffle.partitions", "16") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "4g") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    sc = spark.sparkContext
    print(SEP)
    print("DTSC303 Big Data Computing")
    print(SEP)
    print(f"  App ID  : {sc.applicationId}")
    print(f"  Master  : {sc.master}")
    print(f"  S3 Base : {S3_BASE}")
    print(SEP)
    return spark, sc


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
def load_data(spark):
    print(f"\n{'─'*72}\n  STAGE 1 — Loading data from S3\n{'─'*72}")
    t0 = time.time()

    dfs = {}

    # ─────────────────────────────────────────
    # CORE FILES (AUTO SCHEMA DETECTION)
    # ─────────────────────────────────────────

    dfs["txs_classes"] = spark.read.csv(
        TXS_CLASSES_PATH, header=True, inferSchema=True
    )

    dfs["txs_edgelist"] = spark.read.csv(
        TXS_EDGELIST_PATH, header=True, inferSchema=True
    )

    dfs["wallets_classes"] = spark.read.csv(
        WALLETS_CLASSES_PATH, header=True, inferSchema=True
    )

    dfs["addr_addr"] = spark.read.csv(
        ADDRADDR_PATH, header=True, inferSchema=True
    )

    dfs["addr_tx"] = spark.read.csv(
        ADDRTX_PATH, header=True, inferSchema=True
    )

    dfs["tx_addr"] = spark.read.csv(
        TXADDR_PATH, header=True, inferSchema=True
    )
    try:
        dfs["txaddr_edgelist"] = spark.read.csv(
            DATA_PATH + "TxAddr_edgelist.csv",
            header=True, inferSchema=True
        )
        print("  TxAddr_edgelist.csv LOADED")
    except:
        print("  TxAddr_edgelist.csv missing")

    try:
        dfs["addrtx_edgelist"] = spark.read.csv(
            DATA_PATH + "AddrTx_edgelist.csv",
            header=True, inferSchema=True
        )
        print("  AddrTx_edgelist.csv LOADED")
    except:
        print("  AddrTx_edgelist.csv missing")

    try:
        dfs["wallets_features"] = spark.read.csv(
            DATA_PATH + "wallets_features.csv",
            header=True, inferSchema=True
        )
        print("  wallets_features.csv LOADED")
    except:
        print("  wallets_features.csv missing")

    try:
        dfs["wallets_features_combined"] = spark.read.csv(
            DATA_PATH + "wallets_features_classes_combined.csv",
            header=True, inferSchema=True
        )
        print("  wallets_features_classes_combined.csv LOADED")
    except:
        print("  wallets_features_classes_combined.csv missing")

    try:
        dfs["txs_features"] = spark.read.csv(
            DATA_PATH + "txs_features.csv",
            header=True, inferSchema=True
        )
        print("  txs_features.csv LOADED")
    except:
        print("  txs_features.csv missing")

    print("\n" + "="*72)
    print("SCHEMA CHECK — ALL DATASETS")
    print("="*72)

    for name, df in dfs.items():
        if df is not None:
            print(f"\nDataset: {name}")
            print("-"*50)
            print("Columns:", df.columns)
            df.printSchema()
            print("-"*50)

    print("\n  File                           Rows")
    print("  ────────────────────────────────────")

    for name, df in dfs.items():
        try:
            print(f"  {name:25s} {df.count():,}")
        except:
            print(f"  {name:25s} ERROR")
    
    for name, df in dfs.items():
        if df is not None:
            print(f"\nDataset: {name}")
            print("-"*50)
            print("Columns:", df.columns)
            print("Schema:")
            df.printSchema()
            print("-"*50)
    
    has_features = False
    dfs["txs_features"] = None
    try:
        dfs["txs_features"] = spark.read.csv(
            TXS_FEATURES_PATH, header=True, inferSchema=True)
        print(f"  txs_features.csv  LOADED  ({len(dfs['txs_features'].columns)} cols)")
        has_features = True
    except Exception as e:
        print(f"  txs_features.csv  NOT FOUND — ML stage (PS-4) will be skipped")
        print(f"  Upload to: {TXS_FEATURES_PATH}")

    for key in ["txs_classes","txs_edgelist","wallets_classes","addr_addr"]:
        dfs[key].cache()

    counts = {k: dfs[k].count()
              for k in ["txs_classes","txs_edgelist","wallets_classes",
                        "addr_addr","addr_tx","tx_addr"]}
    print(f"\n  {'File':<22} {'Rows':>12}")
    print(f"  {'─'*36}")
    for name, cnt in counts.items():
        print(f"  {name:<22} {cnt:>12,}")
    print(f"\n  Load time: {time.time()-t0:.2f}s")
    return dfs, has_features

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1.5 — EXPLORATORY DATA ANALYSIS (EDA)
# ══════════════════════════════════════════════════════════════════════════════
def stage_eda(spark, dfs, has_features):
    print(f"\n{'─'*72}\n  STAGE 1.5 — Exploratory Data Analysis (EDA)\n{'─'*72}")
    t0 = time.time()
    import pandas as pd

    # --- A. Missing Values ---
    print("  [EDA A] Checking for Null Values...")
    null_report = []
    for name, df in dfs.items():
        if df is not None:
            # Count nulls in each column and sum them
            null_counts = df.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns]).collect()[0].asDict()
            total_nulls = sum(null_counts.values())
            null_report.append({"dataset": name, "total_rows": df.count(), "null_count": total_nulls})
    
    eda_null_pdf = pd.DataFrame(null_report)
    save_csv(eda_null_pdf, "eda_01_null_check.csv")
    print(eda_null_pdf)

    # --- B. Feature Statistics (Local vs Aggregate) ---
    if has_features:
        print("  [EDA B] Generating Statistical Summary for Features...")
    
        subset_cols = dfs["txs_features"].columns[1:11] 
        stats_df = dfs["txs_features"].select(subset_cols).summary("count", "min", "25%", "50%", "75%", "max", "mean", "stddev")
        stats_pdf = spark_to_csv(stats_df, "eda_02_feature_stats.csv")

        # Feature Distributions 
        feat_sample = dfs["txs_features"].select(subset_cols).limit(10000).toPandas()
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=feat_sample, orient="h", palette="Set2")
        ax.set_title("Distribution of Local Features (First 10 Features)", fontweight="bold")
        ax.set_xlabel("Normalized Value")
        plt.tight_layout()
        save_fig(fig, "18_eda_feature_boxplot.png")

        # --- C. Correlation Analysis ---
        print("  [EDA C] Computing Feature Correlation Matrix...")
        # Correlate first 15 features to avoid memory overhead
        corr_cols = dfs["txs_features"].columns[1:16]
        corr_sample = dfs["txs_features"].select(corr_cols).limit(5000).toPandas()
        corr_matrix = corr_sample.corr()

        # Correlation Heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", center=0, linewidths=0.1)
        ax.set_title("Feature Correlation Heatmap (Subset of 15 features)", fontweight="bold")
        plt.tight_layout()
        save_fig(fig, "19_eda_correlation_heatmap.png")

    print(f"  Stage time: {time.time()-t0:.2f}s")
    
# ══════════════════════════════════════════════════════════════════════════════
# 2. MAPREDUCE PS-1: CLASS DISTRIBUTION  +  Charts 1 & 2
# ══════════════════════════════════════════════════════════════════════════════
def stage_class_distribution(spark, dfs):
    print(f"\n{'─'*72}\n  STAGE 2 — MapReduce: Class Distribution  [PS-1]\n{'─'*72}")
    t0 = time.time()
    import pandas as pd
    label_map = {1:"illicit", 2:"licit", 3:"unknown"}

    # Job A — transaction class counts
    tx_counts   = (dfs["txs_classes"].rdd
                   .map(lambda r: (label_map.get(r["class"],"unknown"), 1))
                   .reduceByKey(lambda a,b: a+b)
                   .sortBy(lambda x: x[1], ascending=False)
                   .collect())
    total_tx    = sum(v for _,v in tx_counts)
    tx_d        = dict(tx_counts)

    # Job B — wallet class counts
    wallet_counts = (dfs["wallets_classes"].rdd
                     .map(lambda r: (label_map.get(r["class"],"unknown"), 1))
                     .reduceByKey(lambda a,b: a+b)
                     .sortBy(lambda x: x[1], ascending=False)
                     .collect())
    total_w     = sum(v for _,v in wallet_counts)
    wl_d        = dict(wallet_counts)

    # Job C — fraud imbalance
    labelled    = {k:v for k,v in tx_d.items() if k!="unknown"}
    ill         = labelled.get("illicit",0)
    lic         = labelled.get("licit",0)

    print(f"\n  [Job A] Transactions:  illicit={ill:,}  licit={lic:,}  unknown={tx_d.get('unknown',0):,}")
    print(f"  [Job B] Wallets:       illicit={wl_d.get('illicit',0):,}  licit={wl_d.get('licit',0):,}")
    print(f"  [Job C] Illicit rate: {ill/(ill+lic)*100:.2f}%   Licit:Illicit = {lic/max(ill,1):.1f}:1")

    cats = ["illicit","licit","unknown"]
    summary = pd.DataFrame({
        "class": cats,
        "tx_count": [tx_d.get(c,0) for c in cats],
        "tx_pct":   [round(tx_d.get(c,0)/total_tx*100,2) for c in cats],
        "wallet_count": [wl_d.get(c,0) for c in cats],
        "wallet_pct":   [round(wl_d.get(c,0)/total_w*100,2) for c in cats],
    })
    save_csv(summary, "01_class_distribution.csv")

    #pie chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Elliptic++ Dataset — Class Distribution", fontsize=14, fontweight="bold")
    colors = [C_ILLICIT, C_LICIT, C_UNKNOWN]
    for ax, (title, d, total) in zip(axes, [
        (f"Transactions (n={total_tx:,})", tx_d, total_tx),
        (f"Wallet Addresses (n={total_w:,})", wl_d, total_w),
    ]):
        vals   = [d.get(c,0) for c in cats]
        labels = ["Illicit (fraud)", "Licit (non-fraud)", "Unknown"]
        wedges, texts, autotexts = ax.pie(
            vals, labels=labels, colors=colors, autopct="%1.1f%%", startangle=140,
            wedgeprops=dict(edgecolor="white", linewidth=1.5))
        for at in autotexts:
            at.set_fontsize(10)
        ax.set_title(title, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "01_class_distribution_pie.png")

    # Grouped bar
    x  = np.arange(3)
    w  = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x-w/2, [tx_d.get(c,0) for c in cats], w,
                   label="Transactions", color=colors, alpha=0.85)
    bars2 = ax.bar(x+w/2, [wl_d.get(c,0) for c in cats], w,
                   label="Wallets", color=colors, alpha=0.45,
                   edgecolor=colors, linewidth=1.5)
    for bar in list(bars1)+list(bars2):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5000,
                f"{int(bar.get_height()):,}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["Illicit","Licit","Unknown"])
    ax.set_ylabel("Count")
    ax.set_title("Class Distribution: Transactions vs Wallets", fontweight="bold")
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{int(v):,}"))
    plt.tight_layout()
    save_fig(fig, "02_class_distribution_bar.png")

    print(f"  Stage time: {time.time()-t0:.2f}s")
    return tx_d, wl_d


# ══════════════════════════════════════════════════════════════════════════════
# 3. MAPREDUCE PS-2: GRAPH DEGREE  +  Charts 3 & 4
# ══════════════════════════════════════════════════════════════════════════════
def stage_graph_degree(spark, dfs):
    print(f"\n{'─'*72}\n  STAGE 3 — MapReduce: Transaction Graph Degree  [PS-2]\n{'─'*72}")
    t0 = time.time()
    import pandas as pd

    edge_rdd   = dfs["txs_edgelist"].rdd
    out_deg    = edge_rdd.map(lambda r: (r["txId1"], 1)).reduceByKey(lambda a,b: a+b)
    in_deg     = edge_rdd.map(lambda r: (r["txId2"], 1)).reduceByKey(lambda a,b: a+b)
    total_deg  = out_deg.union(in_deg).reduceByKey(lambda a,b: a+b)

    top20   = total_deg.map(lambda x: (x[1],x[0])).sortByKey(ascending=False).take(20)
    stats   = total_deg.map(lambda x: float(x[1])).stats()
    cls_rdd = dfs["txs_classes"].rdd.map(lambda r: (r["txId"], r["class"]))
    hub_rdd = spark.sparkContext.parallelize([(txid,deg) for deg,txid in top20])
    labelled_hubs = (hub_rdd.leftOuterJoin(cls_rdd)
                     .map(lambda x: (x[0], x[1][0],
                          {1:"illicit",2:"licit",3:"unknown"}.get(x[1][1] or 3,"unknown")))
                     .sortBy(lambda x: x[1], ascending=False).collect())

    print(f"  Nodes={int(stats.count()):,}  Max={int(stats.max()):,}  "
          f"Mean={stats.mean():.2f}  Stdev={stats.stdev():.2f}")
    print(f"  {'txId':<20} {'Degree':>10} {'Label'}")
    for txid, deg, label in labelled_hubs:
        print(f"  {txid:<20} {deg:>10,}  {label}")

    hubs_pdf = pd.DataFrame(labelled_hubs, columns=["txId","degree","label"])
    save_csv(hubs_pdf, "02_top20_hub_nodes.csv")

    # Horizontal bar, coloured by class
    fig, ax = plt.subplots(figsize=(13, 6))
    ids    = [str(r[0])[-10:] for r in labelled_hubs]
    degs   = [r[1] for r in labelled_hubs]
    colors = [C_ILLICIT if r[2]=="illicit" else C_LICIT if r[2]=="licit" else C_UNKNOWN
              for r in labelled_hubs]
    ax.barh(ids[::-1], degs[::-1], color=colors[::-1], edgecolor="white", linewidth=0.5)
    for i, v in enumerate(degs[::-1]):
        ax.text(v+0.3, i, f"{v:,}", va="center", fontsize=8)
    ax.set_xlabel("Total Degree (in + out)")
    ax.set_title("Top 20 Highest-Degree Transaction Nodes (Potential Mixing Hubs)",
                 fontweight="bold")
    patches = [mpatches.Patch(color=C_ILLICIT, label="Illicit"),
               mpatches.Patch(color=C_LICIT,   label="Licit"),
               mpatches.Patch(color=C_UNKNOWN, label="Unknown")]
    ax.legend(handles=patches, loc="lower right")
    plt.tight_layout()
    save_fig(fig, "03_top_hubs_degree.png")

    # Degree distribution histogram (sampled)
    sample = total_deg.map(lambda x: float(x[1])).takeSample(False, 50000, seed=42)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(sample, bins=60, color=C_BLUE, edgecolor="white", linewidth=0.4, log=True)
    ax.set_xlabel("Node Degree")
    ax.set_ylabel("Frequency (log scale)")
    ax.set_title("Transaction Graph Degree Distribution (50k sample, log scale)",
                 fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "04_degree_distribution_histogram.png")

    print(f"  Stage time: {time.time()-t0:.2f}s")


# ══════════════════════════════════════════════════════════════════════════════
# 4. SPARK SQL PS-3: WALLET NETWORK  +  Charts 5 & 6
# ══════════════════════════════════════════════════════════════════════════════
def stage_wallet_network(spark, dfs):
    print(f"\n{'─'*72}\n  STAGE 4 — Spark SQL: Wallet Network Analysis  [PS-3]\n{'─'*72}")
    t0 = time.time()

    dfs["wallets_classes"].createOrReplaceTempView("wallets")
    dfs["addr_addr"].createOrReplaceTempView("addr_edges")
    dfs["addr_tx"].createOrReplaceTempView("addr_tx_view")
    dfs["tx_addr"].createOrReplaceTempView("tx_addr_view")
    dfs["txs_classes"].createOrReplaceTempView("txs_classes_view")

    # A: edge flow by class pair
    flow_df  = spark.sql("""
        SELECT CASE wi.class WHEN 1 THEN 'illicit' WHEN 2 THEN 'licit' ELSE 'unknown' END AS from_class,
               CASE wo.class WHEN 1 THEN 'illicit' WHEN 2 THEN 'licit' ELSE 'unknown' END AS to_class,
               COUNT(*) AS edge_count
        FROM addr_edges e
        LEFT JOIN wallets wi ON e.input_address  = wi.address
        LEFT JOIN wallets wo ON e.output_address = wo.address
        GROUP BY wi.class, wo.class ORDER BY edge_count DESC
    """)
    flow_df.show()
    flow_pdf = spark_to_csv(flow_df, "03_edge_flow_by_class.csv")

    # B: top illicit wallets by out-degree
    top_ill = spark.sql("""
        SELECT e.input_address, COUNT(e.output_address) AS out_degree,
               COUNT(DISTINCT e.output_address) AS unique_targets
        FROM addr_edges e JOIN wallets w ON e.input_address=w.address AND w.class=1
        GROUP BY e.input_address ORDER BY out_degree DESC LIMIT 15
    """)
    top_ill.show(truncate=True)
    top_ill_pdf = spark_to_csv(top_ill, "04_top_illicit_wallets.csv")

    # C: illicit → licit transfers
    print("  [SQL C] Illicit-to-licit direct transfers:")
    spark.sql("""
        SELECT e.input_address AS illicit_wallet, e.output_address AS licit_wallet,
               COUNT(*) AS transfers
        FROM addr_edges e
        JOIN wallets wi ON e.input_address =wi.address AND wi.class=1
        JOIN wallets wo ON e.output_address=wo.address AND wo.class=2
        GROUP BY e.input_address, e.output_address ORDER BY transfers DESC LIMIT 15
    """).show(truncate=True)

    # Edge flow heatmap
    pivot = flow_pdf.pivot(index="from_class", columns="to_class", values="edge_count").fillna(0)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd",
                linewidths=0.5, linecolor="white", ax=ax, annot_kws={"size":10})
    ax.set_title("Addr-Addr Edge Flow by Class Pair\n(row=sender, col=receiver)",
                 fontweight="bold")
    ax.set_xlabel("Receiver class")
    ax.set_ylabel("Sender class")
    plt.tight_layout()
    save_fig(fig, "05_edge_flow_heatmap.png")

    #Top illicit wallets out-degree
    fig, ax = plt.subplots(figsize=(11, 5))
    short   = [a[-12:]+"…" for a in top_ill_pdf["input_address"]]
    ax.barh(short[::-1], top_ill_pdf["out_degree"][::-1].values,
            color=C_ILLICIT, edgecolor="white", linewidth=0.5)
    for i, (v, u) in enumerate(zip(top_ill_pdf["out_degree"][::-1].values,
                                    top_ill_pdf["unique_targets"][::-1].values)):
        ax.text(v+0.3, i, f"{v:,} ({u:,} unique)", va="center", fontsize=8)
    ax.set_xlabel("Out-degree")
    ax.set_title("Top 15 Illicit Wallets by Out-Degree\n(potential laundering hubs)",
                 fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "06_top_illicit_wallets_outdegree.png")

    print(f"  Stage time: {time.time()-t0:.2f}s")


# ══════════════════════════════════════════════════════════════════════════════
# 5. WINDOW FUNCTIONS PS-5: TEMPORAL ANALYSIS  +  Charts 7–10
# ══════════════════════════════════════════════════════════════════════════════
def stage_temporal_analysis(spark, dfs):
    print(f"\n{'─'*72}\n  STAGE 5 — Window Functions: Temporal Analysis  [PS-5]\n{'─'*72}")
    t0 = time.time()

    if dfs["txs_features"] is not None:
        feat_df = dfs["txs_features"].select(
            F.col("txId").cast(LongType()),
            F.col("Time step").cast(IntegerType()).alias("time_step"))
    else:
        total   = dfs["txs_classes"].count()
        bin_sz  = max(1, total // 49)
        rw      = Window.orderBy("txId")
        feat_df = dfs["txs_classes"] \
            .withColumn("rn", F.row_number().over(rw)) \
            .withColumn("time_step",
                F.least(F.lit(49), ((F.col("rn")-1)/F.lit(bin_sz)).cast(IntegerType())+F.lit(1))) \
            .select("txId","time_step")

    tx_time = feat_df.join(dfs["txs_classes"], on="txId", how="inner")
    tx_time.createOrReplaceTempView("tx_time")

    time_dist = spark.sql("""
        SELECT time_step,
               SUM(CASE WHEN class=1 THEN 1 ELSE 0 END) AS illicit,
               SUM(CASE WHEN class=2 THEN 1 ELSE 0 END) AS licit,
               SUM(CASE WHEN class=3 THEN 1 ELSE 0 END) AS unknown,
               COUNT(*) AS total
        FROM tx_time GROUP BY time_step ORDER BY time_step
    """)

    cum_win  = Window.orderBy("time_step").rowsBetween(Window.unboundedPreceding, Window.currentRow)
    roll_win = Window.orderBy("time_step").rowsBetween(-2, 0)
    enriched = time_dist \
        .withColumn("cum_illicit", F.sum("illicit").over(cum_win)) \
        .withColumn("cum_total",   F.sum("total").over(cum_win)) \
        .withColumn("illicit_rate_pct",
                    F.round(F.col("cum_illicit")/F.col("cum_total")*100,3)) \
        .withColumn("rolling_3step", F.round(F.avg("illicit").over(roll_win),2))

    time_pdf = spark_to_csv(enriched, "06_temporal_fraud_distribution.csv")
    steps    = time_pdf["time_step"].tolist()

    # Stacked bar per time step
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(steps, time_pdf["unknown"], color=C_UNKNOWN, label="Unknown (unlabelled)")
    ax.bar(steps, time_pdf["licit"],   color=C_LICIT,   label="Licit (non-fraud)",
           bottom=time_pdf["unknown"])
    ax.bar(steps, time_pdf["illicit"], color=C_ILLICIT, label="Illicit (fraud)",
           bottom=(time_pdf["unknown"]+time_pdf["licit"]))
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Number of Transactions")
    ax.set_title("Transaction Distribution per Time Step by Class", fontweight="bold")
    ax.set_xticks(steps[::2])
    ax.legend(loc="upper right")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{int(v):,}"))
    plt.tight_layout()
    save_fig(fig, "07_stacked_bar_per_timestep.png")

    #Line chart with rolling avg and train/test split line
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(steps, time_pdf["illicit"], alpha=0.2, color=C_ILLICIT)
    ax.plot(steps, time_pdf["illicit"],     color=C_ILLICIT, lw=2, label="Illicit count")
    ax.plot(steps, time_pdf["licit"],       color=C_LICIT,   lw=2, label="Licit count")
    ax.plot(steps, time_pdf["rolling_3step"], color="black", lw=1.5,
            ls="--", label="3-step rolling avg (illicit)")
    ax.axvline(x=TRAIN_CUTOFF, color="navy", ls=":", lw=1.5,
               label=f"Train/Test split (step {TRAIN_CUTOFF})")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Transaction Count")
    ax.set_title("Temporal Fraud Trends Across 49 Time Steps", fontweight="bold")
    ax.legend()
    ax.set_xticks(steps[::2])
    plt.tight_layout()
    save_fig(fig, "08_temporal_trends_line.png")

    #Cumulative illicit rate
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(steps, time_pdf["illicit_rate_pct"], color=C_PURPLE, lw=2)
    ax.fill_between(steps, time_pdf["illicit_rate_pct"], alpha=0.15, color=C_PURPLE)
    ax.axvline(x=TRAIN_CUTOFF, color="navy", ls=":", lw=1.5,
               label=f"Train/Test split (step {TRAIN_CUTOFF})")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Cumulative Illicit Rate (%)")
    ax.set_title("Cumulative Illicit Transaction Rate Over Time", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    save_fig(fig, "09_cumulative_illicit_rate.png")

    # Fraud propagation: neighbour illicit ratio
    ec = dfs["txs_edgelist"] \
        .join(dfs["txs_classes"].withColumnRenamed("class","c1").withColumnRenamed("txId","x1"),
              F.col("txId1")==F.col("x1"),"left").drop("x1") \
        .join(dfs["txs_classes"].withColumnRenamed("class","c2").withColumnRenamed("txId","x2"),
              F.col("txId2")==F.col("x2"),"left").drop("x2")

    nb = ec.groupBy("txId1","c1").agg(
        F.count("txId2").alias("total_nb"),
        F.sum(F.when(F.col("c2")==1,1).otherwise(0)).alias("illicit_nb")
    ).withColumn("illicit_nb_ratio", F.round(F.col("illicit_nb")/F.col("total_nb"),4))

    nb_pdf = spark_to_csv(
        nb.filter(F.col("c1")==1).orderBy(F.desc("illicit_nb_ratio")).limit(500),
        "07_fraud_propagation_neighbour_ratio.csv")

    # Fraud propagation histogram
    if len(nb_pdf) > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(nb_pdf["illicit_nb_ratio"], bins=30, color=C_ILLICIT,
                edgecolor="white", linewidth=0.4)
        ax.set_xlabel("Fraction of illicit neighbours")
        ax.set_ylabel("Count of illicit transactions")
        ax.set_title("Fraud Propagation — Illicit Neighbour Ratio\n"
                     "EASY cases cluster at 1.0, HARD cases cluster at 0.0",
                     fontweight="bold")
        plt.tight_layout()
        save_fig(fig, "10_fraud_propagation_hist.png")

    print(f"  Stage time: {time.time()-t0:.2f}s")
    return time_pdf


# ══════════════════════════════════════════════════════════════════════════════
# 6. ML PS-4: FRAUD DETECTION  +  Charts 11–13
# ══════════════════════════════════════════════════════════════════════════════
def stage_ml_fraud(spark, dfs, has_features):
    print(f"\n{'─'*72}\n  STAGE 6 — ML: Fraud Detection  [PS-4]\n{'─'*72}")
    if not has_features:
        print(f"  SKIPPED — upload txs_features.csv to {S3_BASE}/")
        return None
    t0 = time.time()
    import pandas as pd

    ml_df = dfs["txs_features"] \
        .join(dfs["txs_classes"].withColumnRenamed("class","label_raw"), on="txId", how="inner") \
        .dropna() \
        .filter(F.col("label_raw").isin([ILLICIT, LICIT])) \
        .withColumn("label", (F.col("label_raw")==ILLICIT).cast(IntegerType()))

    meta   = {"txId","Time step","label_raw","label"}
    f_cols = [c for c in ml_df.columns if c not in meta]
    assembler = VectorAssembler(inputCols=f_cols, outputCol="features_raw", handleInvalid="skip")
    scaler    = StandardScaler(inputCol="features_raw", outputCol="features",
                               withMean=True, withStd=True)

    print("\n  Using 5-Fold Cross Validation...")

    evaluator = BinaryClassificationEvaluator(
        labelCol="label",
        metricName="areaUnderPR"
    )

    rf = RandomForestClassifier(
        labelCol="label",
        featuresCol="features",
        seed=42
    )

    pipeline = Pipeline(stages=[assembler, scaler, rf])

    paramGrid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [50, 100]) \
        .addGrid(rf.maxDepth, [5, 10]) \
        .build()

    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator,
        numFolds=5,
        parallelism=2
    )

    all_results    = {}
    rf_importances = None

    for name, clf in [
        ("Random Forest",      RandomForestClassifier(featuresCol="features", labelCol="label",
                                                      numTrees=100, maxDepth=10, seed=42)),
        ("Logistic Regression",LogisticRegression(featuresCol="features", labelCol="label",
                                                  maxIter=100, regParam=0.01)),
    ]:
        print(f"\n  ── {name} ──")
        pipe  = Pipeline(stages=[assembler, scaler, clf])
        cv_model = cv.fit(ml_df)
        preds = cv_model.transform(ml_df)
        score = evaluator.evaluate(preds)
        print(f"  Cross-Validated PR AUC: {score:.4f}")
        print("  Best Model:", cv_model.bestModel.stages[-1])

        acc = MulticlassClassificationEvaluator(labelCol="label",metricName="accuracy").evaluate(preds)
        f1  = MulticlassClassificationEvaluator(labelCol="label",metricName="f1").evaluate(preds)
        auc = BinaryClassificationEvaluator(labelCol="label",metricName="areaUnderPR").evaluate(preds)
        tp  = preds.filter("label=1 AND prediction=1").count()
        fp  = preds.filter("label=0 AND prediction=1").count()
        fn  = preds.filter("label=1 AND prediction=0").count()
        tn  = preds.filter("label=0 AND prediction=0").count()
        prec = tp/(tp+fp) if tp+fp>0 else 0
        rec  = tp/(tp+fn) if tp+fn>0 else 0

        print(f"    Accuracy={acc:.4f}  F1={f1:.4f}  AUC-PR={auc:.4f}")
        print(f"    Precision={prec:.4f}  Recall={rec:.4f}")
        print(f"    TP:{tp:,}  FP:{fp:,}  FN:{fn:,}  TN:{tn:,}")
        all_results[name] = dict(acc=acc,f1=f1,auc=auc,prec=prec,rec=rec,
                                  tp=tp,fp=fp,fn=fn,tn=tn)

        if name == "Random Forest":
            rf_importances = sorted(
                zip(f_cols, cv_model.bestModel.stages[-1].featureImportances.toArray()),
                key=lambda x: x[1], reverse=True)
            print("  Top 10 features:")
            for feat, imp in rf_importances[:10]:
                print(f"    {feat:<35} {imp:.6f}")

        # Save predictions as single CSV
        safe  = name.replace(" ","_").lower()
        preds.select("txId","label","prediction") \
             .coalesce(1).write.mode("overwrite") \
             .option("header","true") \
             .csv(f"{OUTPUT_BASE}/csv/predictions_{safe}")

    # Model comparison bar
    metrics = ["acc","f1","auc","prec","rec"]
    mlabels = ["Accuracy","F1","AUC-PR","Precision","Recall"]
    x, w   = np.arange(len(metrics)), 0.3
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (mname, r) in enumerate(all_results.items()):
        bars = ax.bar(x+i*w-w/2, [r[m] for m in metrics], w, label=mname, alpha=0.85)
        for bar, val in zip(bars, [r[m] for m in metrics]):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(mlabels)
    ax.set_ylim(0, 1.12); ax.set_ylabel("Score")
    ax.set_title("Model Performance — Fraud Detection\n"
                 "(Temporal split: train steps 1-34, test steps 35-49)", fontweight="bold")
    ax.legend(); ax.axhline(0.5, color="grey", ls=":", lw=0.8)
    plt.tight_layout()
    save_fig(fig, "11_model_comparison.png")

    # Confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (mname, r) in zip(axes, all_results.items()):
        cm = np.array([[r["tn"],r["fp"]],[r["fn"],r["tp"]]])
        sns.heatmap(cm, annot=True, fmt=",", cmap="Blues", ax=ax,
                    xticklabels=["Pred:Licit","Pred:Illicit"],
                    yticklabels=["True:Licit","True:Illicit"],
                    linewidths=0.5, linecolor="white")
        ax.set_title(f"{mname}\nF1={r['f1']:.3f}  AUC-PR={r['auc']:.3f}", fontweight="bold")
    fig.suptitle("Confusion Matrices — Fraud Detection", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "12_confusion_matrices.png")

    #Feature importances
    if rf_importances:
        top20  = rf_importances[:20]
        fnames = [f[0] for f in top20]
        fimps  = [f[1] for f in top20]
        colors = [C_ILLICIT if "Local" in n else C_BLUE if "Aggregate" in n else C_PURPLE
                  for n in fnames]
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(fnames[::-1], fimps[::-1], color=colors[::-1], edgecolor="white", lw=0.4)
        for i, v in enumerate(fimps[::-1]):
            ax.text(v+0.0002, i, f"{v:.4f}", va="center", fontsize=8)
        ax.set_xlabel("Importance (Gini)")
        ax.set_title("Top 20 Feature Importances — Random Forest\n"
                     "Red=Local  Blue=Aggregate  Purple=Augmented", fontweight="bold")
        plt.tight_layout()
        save_fig(fig, "13_feature_importances.png")

        imp_pdf = pd.DataFrame(rf_importances, columns=["feature","importance"])
        save_csv(imp_pdf, "08_feature_importances.csv")

    comp_pdf = pd.DataFrame(
        [(k,v["acc"],v["f1"],v["auc"],v["prec"],v["rec"]) for k,v in all_results.items()],
        columns=["model","accuracy","f1","auc_pr","precision","recall"])
    save_csv(comp_pdf, "09_model_comparison.csv")
    print(f"  Stage time: {time.time()-t0:.2f}s")
    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# 7. WALLET CLUSTERING  +  Charts 14 & 15
# ══════════════════════════════════════════════════════════════════════════════
def stage_wallet_clustering(spark, dfs):
    print(f"\n{'─'*72}\n  STAGE 7 — ML: Wallet Clustering (KMeans k=4)\n{'─'*72}")
    t0 = time.time()

    out_deg = dfs["addr_addr"].groupBy("input_address").agg(
        F.count("output_address").alias("out_degree"),
        F.countDistinct("output_address").alias("unique_out")
    ).withColumnRenamed("input_address","address")

    in_deg = dfs["addr_addr"].groupBy("output_address").agg(
        F.count("input_address").alias("in_degree"),
        F.countDistinct("input_address").alias("unique_in")
    ).withColumnRenamed("output_address","address")

    tx_sent = dfs["addr_tx"].groupBy("input_address").agg(
        F.countDistinct("txId").alias("txs_sent")
    ).withColumnRenamed("input_address","address")

    tx_recv = dfs["tx_addr"].groupBy("output_address").agg(
        F.countDistinct("txId").alias("txs_received")
    ).withColumnRenamed("output_address","address")

    wf = (out_deg.join(in_deg,  on="address",how="outer")
                 .join(tx_sent, on="address",how="outer")
                 .join(tx_recv, on="address",how="outer")
                 .join(dfs["wallets_classes"], on="address",how="left")
                 .fillna(0))
    wf.cache()

    f_cols = ["out_degree","in_degree","unique_out","unique_in","txs_sent","txs_received"]
    from pyspark.ml.clustering import KMeans
    from pyspark.ml.evaluation import ClusteringEvaluator

    print("\n  Selecting optimal k using silhouette score...")

    assembler = VectorAssembler(inputCols=f_cols, outputCol="features_raw", handleInvalid="skip")
    scaler    = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)

    base_pipeline = Pipeline(stages=[assembler, scaler])
    wf_transformed = base_pipeline.fit(wf).transform(wf)

    evaluator = ClusteringEvaluator(featuresCol="features")

    k_values = [2, 3, 4, 5, 6]
    scores = {}

    for k in k_values:
        kmeans = KMeans(featuresCol="features", k=k, seed=42)
        cv_model = kmeans.fit(wf_transformed)
        preds = cv_model.transform(wf_transformed)
        score = evaluator.evaluate(preds)

        scores[k] = score
        print(f"  k={k} → silhouette={score:.4f}")

    best_k = max(scores, key=scores.get)
    print(f"\n  BEST k selected: {best_k}")

    clustered = cv_model.transform(wf_transformed)
    # ─────────────────────────────────────────
    # CLUSTER SCATTER VISUALIZATION (PCA)
    # ─────────────────────────────────────────
    from pyspark.ml.feature import PCA
    
    print("\n  Generating cluster scatter plot...")
    
    # Reduce to 2D
    pca = PCA(k=2, inputCol="features", outputCol="pca_features")
    pca_model = pca.fit(clustered)
    pca_result = pca_model.transform(clustered)
    
    # Convert to pandas (only needed columns)
    pdf = pca_result.select("pca_features", "prediction").toPandas()
    
    # Extract x, y
    pdf["x"] = pdf["pca_features"].apply(lambda v: float(v[0]))
    pdf["y"] = pdf["pca_features"].apply(lambda v: float(v[1]))
    
    # Plot
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(8,6))
    
    for c in sorted(pdf["prediction"].unique()):
        subset = pdf[pdf["prediction"] == c]
        plt.scatter(subset["x"], subset["y"], label=f"Cluster {c}", alpha=0.6)
    
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("Wallet Clusters (2D Projection)")
    plt.legend()
    plt.tight_layout()
    
    save_fig(plt.gcf(), "17_cluster_scatter.png")
    plt.show()
    sil = ClusteringEvaluator(featuresCol="features").evaluate(clustered)
    print(f"  Silhouette score (k=4): {sil:.4f}")

    profile_df = clustered.groupBy("prediction").agg(
        F.count("*").alias("wallet_count"),
        F.round(F.avg("out_degree"),2).alias("avg_out_deg"),
        F.round(F.avg("in_degree"),2).alias("avg_in_deg"),
        F.round(F.avg("txs_sent"),2).alias("avg_txs_sent"),
        F.round(F.avg("txs_received"),2).alias("avg_txs_rcvd"),
        F.sum(F.when(F.col("class")==1,1).otherwise(0)).alias("illicit_count"),
        F.sum(F.when(F.col("class")==2,1).otherwise(0)).alias("licit_count"),
    ).orderBy("prediction")
    print("\n  Interpreting clusters...")

    profile_df = profile_df.withColumn(
        "cluster_type",
        F.when(F.col("avg_out_deg") > 50, "High Activity Hub")
         .when(F.col("avg_in_deg") > F.col("avg_out_deg"), "Receiver Wallet")
         .when(F.col("avg_out_deg") > F.col("avg_in_deg"), "Sender Wallet")
         .otherwise("Low Activity")
    )

    profile_df.show(truncate=False)

    profile_df.show()
    prof_pdf = spark_to_csv(profile_df, "10_wallet_cluster_profiles.csv")

    cnames = [f"Cluster {i}" for i in prof_pdf["prediction"]]
    ccolors = [C_BLUE, C_LICIT, C_UNKNOWN, C_PURPLE]

    # Cluster profiles grouped bar
    metrics = ["avg_out_deg","avg_in_deg","avg_txs_sent","avg_txs_rcvd"]
    mlabels = ["Avg Out-Deg","Avg In-Deg","Avg Txs Sent","Avg Txs Rcvd"]
    x, w    = np.arange(len(metrics)), 0.2
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (cn, cc) in enumerate(zip(cnames, ccolors)):
        vals = [float(prof_pdf[m].iloc[i]) for m in metrics]
        ax.bar(x+i*w-1.5*w/2, vals, w, label=cn, color=cc, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(mlabels)
    ax.set_ylabel("Average Value")
    ax.set_title("Wallet Cluster Profiles (KMeans k=4)", fontweight="bold")
    ax.legend(); plt.tight_layout()
    save_fig(fig, "14_wallet_cluster_profiles.png")

    # Illicit vs licit per cluster
    x, w = np.arange(len(cnames)), 0.3
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x-w/2, prof_pdf["illicit_count"].values, w, label="Illicit", color=C_ILLICIT, alpha=0.85)
    ax.bar(x+w/2, prof_pdf["licit_count"].values,   w, label="Licit",   color=C_LICIT,   alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(cnames)
    ax.set_ylabel("Wallet Count")
    ax.set_title("Illicit vs Licit Wallet Counts per Cluster", fontweight="bold")
    ax.legend(); plt.tight_layout()
    save_fig(fig, "15_cluster_illicit_vs_licit.png")

    print(f"  Stage time: {time.time()-t0:.2f}s")

    plt.figure(figsize=(6,4))
    plt.plot(list(scores.keys()), list(scores.values()), marker='o')

    plt.xlabel("k")
    plt.ylabel("Silhouette Score")
    plt.title("Choosing Optimal k")

    plt.tight_layout()
    save_fig(plt.gcf(), "13_k_selection.png")
    plt.show()

    # Chart — Cluster behavior map
    pdf = profile_df.toPandas()

    plt.figure(figsize=(7,5))
    plt.scatter(pdf["avg_out_deg"], pdf["avg_in_deg"])

    for i, txt in enumerate(pdf["cluster_type"]):
        plt.annotate(txt, (pdf["avg_out_deg"][i], pdf["avg_in_deg"][i]))

    plt.xlabel("Avg Out Degree")
    plt.ylabel("Avg In Degree")
    plt.title("Cluster Behavior Map")

    plt.tight_layout()
    save_fig(plt.gcf(), "16_cluster_behavior_map.png")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 8. CASE ANALYSIS  +  Chart 16
# ══════════════════════════════════════════════════════════════════════════════
def stage_case_analysis(spark, dfs):
    print(f"\n{'─'*72}\n  STAGE 8 — Case Analysis: Easy / Hard / Average Fraud\n{'─'*72}")
    t0 = time.time()

    ec = dfs["txs_edgelist"] \
        .join(dfs["txs_classes"].withColumnRenamed("class","c1").withColumnRenamed("txId","x1"),
              F.col("txId1")==F.col("x1"),"left").drop("x1") \
        .join(dfs["txs_classes"].withColumnRenamed("class","c2").withColumnRenamed("txId","x2"),
              F.col("txId2")==F.col("x2"),"left").drop("x2")

    cases = ec.filter(F.col("c1")==1) \
        .groupBy("txId1") \
        .agg(F.count("txId2").alias("total_nb"),
             F.sum(F.when(F.col("c2")==1,1).otherwise(0)).alias("illicit_nb"),
             F.sum(F.when(F.col("c2")==2,1).otherwise(0)).alias("licit_nb")) \
        .withColumn("illicit_ratio", F.round(F.col("illicit_nb")/F.col("total_nb"),4)) \
        .withColumn("case_type",
            F.when(F.col("illicit_ratio")==1.0,"EASY")
             .when(F.col("illicit_ratio")==0.0,"HARD")
             .otherwise("AVERAGE"))

    summary = cases.groupBy("case_type").agg(
        F.count("*").alias("count"),
        F.round(F.avg("total_nb"),2).alias("avg_neighbours"),
        F.round(F.avg("illicit_ratio"),4).alias("avg_illicit_ratio"),
    ).orderBy(F.desc("count"))
    summary.show()
    sum_pdf = spark_to_csv(summary, "11_case_analysis_summary.csv")

    #Case analysis pie + bar
    case_colors = {"EASY":C_LICIT,"HARD":C_ILLICIT,"AVERAGE":C_UNKNOWN}
    fig, axes   = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Fraud Case Analysis — Easy / Hard / Average", fontweight="bold", fontsize=13)

    ct     = sum_pdf.set_index("case_type")["count"]
    cpie   = [case_colors.get(k,"grey") for k in ct.index]
    axes[0].pie(ct.values, labels=ct.index, colors=cpie, autopct="%1.1f%%", startangle=140,
                wedgeprops=dict(edgecolor="white",linewidth=1.5))
    axes[0].set_title("Case Type Distribution\n(illicit txns with edges)")

    ct_types = sum_pdf["case_type"].tolist()
    ct_nb    = sum_pdf["avg_neighbours"].tolist()
    cbar     = [case_colors.get(k,"grey") for k in ct_types]
    bars = axes[1].bar(ct_types, ct_nb, color=cbar, edgecolor="white", lw=0.5)
    for bar, val in zip(bars, ct_nb):
        axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                     f"{val:.1f}", ha="center", va="bottom", fontsize=10)
    axes[1].set_ylabel("Avg Neighbour Count")
    axes[1].set_title("Avg Neighbours per Case Type\nGreen=EASY  Red=HARD  Orange=AVERAGE")

    plt.tight_layout()
    save_fig(fig, "16_case_analysis.png")
    print(f"  Stage time: {time.time()-t0:.2f}s")


# ══════════════════════════════════════════════════════════════════════════════
# 9. YARN PERFORMANCE  +  Chart 17
# ══════════════════════════════════════════════════════════════════════════════
def stage_yarn_performance(spark, pipeline_start):
    print(f"\n{'─'*72}\n  STAGE 9 — YARN Performance Summary  [PS-6]\n{'─'*72}")
    sc   = spark.sparkContext
    conf = spark.conf
    total = time.time() - pipeline_start

    print(f"\n  App ID              : {sc.applicationId}")
    print(f"  Master              : {sc.master}")
    print(f"  Default Parallelism : {sc.defaultParallelism}")
    print(f"  Executor Memory     : {conf.get('spark.executor.memory','N/A')}")
    print(f"  Driver Memory       : {conf.get('spark.driver.memory','N/A')}")
    print(f"  Shuffle Partitions  : {conf.get('spark.sql.shuffle.partitions')}")
    print(f"  Total Pipeline Time : {total:.1f}s  ({total/60:.1f} min)")

    # Chart 17 — Pipeline overview
    stages = ["1. Data Loading","2. Class Dist (MapReduce)","3. Graph Degree (MapReduce)",
              "4. Wallet Network (SQL)","5. Temporal Analysis (Window)",
              "6. ML Fraud Detection","7. Wallet Clustering","8. Case Analysis"]
    ps_lbl = ["—","PS-1","PS-2","PS-3","PS-5","PS-4","PS-4","PS-4"]
    colors = [C_BLUE,C_LICIT,C_LICIT,C_PURPLE,C_UNKNOWN,C_ILLICIT,C_ILLICIT,C_ILLICIT]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.barh(stages[::-1], [1]*len(stages), color=colors[::-1], edgecolor="white", alpha=0.75)
    for i, ps in enumerate(ps_lbl[::-1]):
        ax.text(0.02, i, ps, va="center", fontsize=9, fontweight="bold", color="white")
    ax.set_xlim(0,1.2); ax.set_xticks([])
    ax.set_title("Elliptic++ Pipeline — Stage Overview  "
                 "(Blue=Infra  Green=MapReduce  Purple=SQL  Orange=Window  Red=ML)",
                 fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "17_pipeline_overview.png")

    print(f"""
  YARN commands (run in EMR terminal):
    yarn application -status {sc.applicationId}
    yarn logs -applicationId {sc.applicationId}

  Spark UI (SSH tunnel):
    ssh -i emrpk.pem -L 4040:localhost:4040 hadoop@<EMR_DNS>
    → http://localhost:4040

  Download outputs:
    aws s3 sync "{OUTPUT_BASE}/csv/"    ./output_csv/
    aws s3 sync "{OUTPUT_BASE}/charts/" ./output_charts/
""")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    t_start = time.time()

    spark, sc    = create_spark_session()
    dfs, has_feat = load_data(spark)
    stage_eda(spark, dfs, has_feat) 
    stage_class_distribution(spark, dfs)
    stage_graph_degree(spark, dfs)
    stage_wallet_network(spark, dfs)
    stage_temporal_analysis(spark, dfs)
    ml_results = stage_ml_fraud(spark, dfs, has_feat)
    stage_wallet_clustering(spark, dfs)
    stage_case_analysis(spark, dfs)
    stage_yarn_performance(spark, t_start)

    total = time.time() - t_start
    print(f"\n{SEP}")
    print(f"  PIPELINE COMPLETE — {total:.1f}s  ({total/60:.1f} min)")
    print(f"  CSV tables → {OUTPUT_BASE}/csv/")
    print(f"  PNG charts → {OUTPUT_BASE}/charts/")
    print(SEP)

    spark.stop()


if __name__ == "__main__":
    main()
