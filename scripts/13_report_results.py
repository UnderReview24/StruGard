"""Aggregate real saved results and export the five required scientific figures."""
import _bootstrap
import argparse
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.logging import save_json
from src.utils.config import load_config


NAMES = {"sequential":"Sequential FT","random_replay":"Random replay",
         "structural_replay":"Structural replay","ewc":"EWC","fixed_5":"Fixed 5 experts",
         "rae":"R/A/E","joint":"Joint oracle","always_expand":"Always expand",
         "fixed_2":"Fixed 2 experts","fixed_3":"Fixed 3 experts"}
PRIMARY = ["sequential","random_replay","structural_replay","ewc","fixed_5","rae","joint"]
COLORS = dict(zip(PRIMARY,["#4C78A8","#F58518","#59A14F","#B279A2","#8C8C8C","#D1495B","#252525"]))


def save_figure(fig,path):
    fig.savefig(path.with_suffix(".pdf"),bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"),dpi=170,bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root",default="outputs/pilot")
    p.add_argument("--shift",default="outputs/shift_validation")
    p.add_argument("--partial",action="store_true",help="Report completed runs and explicitly list unfinished methods")
    a = p.parse_args()
    root = Path(a.root)
    out = root/"summary"
    out.mkdir(parents=True,exist_ok=True)
    completed_primary,unfinished = [],[]
    for method in PRIMARY:
        status = root/method/"status.json"
        if not status.exists() or json.loads(status.read_text())["status"]!="passed":
            if not a.partial:
                raise SystemExit(f"Cannot report a completed benchmark: {method} is not finished")
            unfinished.append(method)
            continue
        if not (root/method/"audit.json").exists() or json.loads((root/method/"audit.json").read_text())["status"]!="passed":
            raise SystemExit(f"Missing output audit: {method}")
        completed_primary.append(method)
    finals,stages,matrices,histories = [],[],[],[]
    for path in sorted(root.glob("*/final_results.csv")):
        if path.parent==out:
            continue
        name = path.parent.name
        if not (path.parent/"audit.json").exists():
            raise ValueError(f"Missing audit for {name}")
        final = pd.read_csv(path)
        finals.append(final)
        stage = pd.read_csv(path.parent/"stage_metrics.csv")
        stages.append(stage)
        matrix = pd.read_csv(path.parent/"continual_matrix.csv")
        matrix = matrix.melt(id_vars="stage",var_name="test_domain",value_name="mae")
        matrix.insert(0,"method",name)
        matrices.append(matrix)
        history = path.parent/"expert_history.csv"
        if history.exists():
            h = pd.read_csv(history)
            h.insert(0,"method",name)
            histories.append(h)
    final = pd.concat(finals,ignore_index=True)
    all_stages = pd.concat(stages,ignore_index=True)
    final.to_csv(out/"final_results.csv",index=False)
    all_stages.to_csv(out/"stage_metrics.csv",index=False)
    pd.concat(matrices).to_csv(out/"continual_matrix.csv",index=False)
    all_stages[["method","stage","forgetting"]].to_csv(out/"forgetting.csv",index=False)
    all_stages[["method","stage","num_experts","base_parameters","plastic_parameters","total_parameters","growth_percent"]].to_csv(out/"parameter_growth.csv",index=False)
    pd.concat(histories).to_csv(out/"expert_history.csv",index=False)
    for name in ["novelty_error.csv","novelty_error_bins.csv","novelty_error_pre_adaptation.csv",
                 "novelty_error_pre_adaptation_bins.csv"]:
        shutil.copy2(root/"rae"/name,out/name)
    reference = final.set_index("method")
    shift = json.loads((Path(a.shift)/"gate.json").read_text())
    novel = json.loads((root/"rae/novelty_error_pre_adaptation.json").read_text())
    criteria = dict(structural_shift_verified=shift["status"]=="passed",
        positive_single_run_forgetting=float(reference.at["sequential","forgetting"])>0,
        novelty_error_relationship=novel["positive_relationship"],
        rae_better_than_random_replay=float(reference.at["rae","average_seen_mae"])<float(reference.at["random_replay","average_seen_mae"]),
        dynamic_fewer_experts_than_always_expand=int(reference.at["rae","num_experts"])<int(reference.at["always_expand","num_experts"]))
    criteria["success_criteria_met"] = all(criteria.values())
    criteria.update(sample_count=5000,model_runs=len(final),unfinished_methods=unfinished,
        scale_30000_executed=False,full_dataset_executed=False,
        limitation="One seed and three merged crystal-system domains; EWC/adapter settings are untuned.",
        interpretation="A positive pilot forgetting value is descriptive, not a multi-seed statistical claim.")
    save_json(out/"scientific_status.json",criteria)

    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,
                         "pdf.fonttype":42,"ps.fonttype":42,"font.family":"DejaVu Sans"})
    markers = ["o","s","^","D","v","P","X"]
    for field,filename,ylabel in [
        ("average_seen_mae","average_mae_vs_stage","Average seen-domain MAE (eV/atom)"),
        ("forgetting","forgetting_vs_stage","Error-based forgetting (eV/atom)")]:
        fig,ax = plt.subplots(figsize=(8.6,4.3))
        for method,marker in zip(completed_primary,markers):
            rows = all_stages[all_stages.method==method].sort_values("stage")
            ax.plot(rows.stage,rows[field],label=NAMES[method],color=COLORS[method],marker=marker,
                    linewidth=2.2 if method=="rae" else 1.6,markersize=5,linestyle="--" if method=="joint" else "-")
        ax.set(xlabel="Stream stage",ylabel=ylabel,xticks=sorted(all_stages.stage.unique()))
        ax.grid(alpha=.2)
        ax.legend(loc="center left",bbox_to_anchor=(1.01,.5),frameon=False,fontsize=9)
        fig.tight_layout()
        save_figure(fig,out/filename)

    fig,axes = plt.subplots(1,2,figsize=(10.2,4.1),sharey=True)
    for ax,name,title in zip(axes,["novelty_error_pre_adaptation.csv","novelty_error.csv"],["At domain arrival","After final adaptation"]):
        table = pd.read_csv(out/name)
        ax.scatter(table.novelty_score,table.absolute_error,s=11,alpha=.32,color="#4C78A8",edgecolors="none")
        bins = pd.qcut(table.novelty_score,5,labels=False,duplicates="drop")
        means = table.assign(bin=bins).groupby("bin").agg(score=("novelty_score","mean"),mae=("absolute_error","mean"))
        ax.plot(means.score,means.mae,"o-",color="#D1495B",linewidth=2,label="Quintile MAE")
        rho = table.novelty_score.corr(table.absolute_error,method="spearman")
        ax.set(title=f"{title}\nSpearman rho = {rho:.3f}, n = {len(table)}",xlabel="Nearest-prototype novelty score")
        ax.grid(alpha=.15)
        ax.legend(frameon=False,fontsize=8)
    axes[0].set_ylabel("Absolute formation-energy error (eV/atom)")
    fig.tight_layout()
    save_figure(fig,out/"novelty_vs_error")

    fig,ax = plt.subplots(figsize=(7.4,4.1))
    for method,color,marker in [("rae","#D1495B","o"),("always_expand","#B279A2","s"),
                                ("fixed_2","#4C78A8","^"),("fixed_3","#F58518","D"),("fixed_5","#8C8C8C","v")]:
        rows = all_stages[all_stages.method==method].sort_values("stage")
        ax.plot(rows.stage,rows.num_experts,label=NAMES[method],color=color,marker=marker,linewidth=1.8)
    ax.set(xlabel="Stream stage",ylabel="Number of experts",xticks=sorted(all_stages.stage.unique()),
           yticks=range(1,int(all_stages.num_experts.max())+1))
    ax.grid(alpha=.2)
    ax.legend(loc="center left",bbox_to_anchor=(1.01,.5),frameon=False,fontsize=9)
    fig.tight_layout()
    save_figure(fig,out/"number_of_experts_vs_stage")

    distance = pd.read_csv(Path(a.shift)/"structural_domain_distance.csv",index_col=0)
    config = load_config(root/"rae/config.yaml")
    domain_report = pd.read_csv(Path(config["domains"]["assignments"]).with_suffix(".summary.csv")).set_index("domain_id")
    labels = [f"D{i+1}\n"+str(domain_report.at[int(domain),"domain_name"]).replace("_"," +\n") for i,domain in enumerate(distance.index)]
    fig,ax = plt.subplots(figsize=(6.7,5.7))
    values = distance.to_numpy()
    view = ax.imshow(values,cmap="Blues")
    for i in range(len(values)):
        for j in range(len(values)):
            ax.text(j,i,f"{values[i,j]:.2f}",ha="center",va="center",color="white" if values[i,j]>.6*values.max() else "#222222")
    ax.set(xticks=range(len(values)),yticks=range(len(values)),xticklabels=labels,yticklabels=labels)
    ax.tick_params(axis="both",length=0,pad=10,labelsize=9)
    fig.colorbar(view,ax=ax,shrink=.72,label="Distance between normalized geometry centers")
    fig.tight_layout()
    save_figure(fig,out/"structural_domain_matrix")

    table = final[["method","average_seen_mae","forgetting","num_experts","total_parameters","replay_memory"]]
    text = "# Continual Crystal - executed pilot\n\n"
    text += "5000 structures, three explicitly merged crystal-system domains, seed 42. Results below are measured held-out errors in eV/atom.\n\n"
    text += table.to_markdown(index=False,floatfmt=".6f")+"\n\n"
    text += "The implementation and output audits passed. The default R/A/E method did **not** beat Random Replay; the model-superiority success criterion failed.\n\n"
    if unfinished:
        text += "Training was stopped at the user's request. Unfinished methods: "+", ".join(unfinished)+". No held-out result is reported for these methods. The 30000-sample and full-data experiments were not started.\n\n"
    route_counts = []
    for path in sorted((root/"rae").glob("stage_*_training_routes.csv")):
        routes = pd.read_csv(path)
        route_counts.append(f"stage {path.name.split('_')[1]}: {(routes.decision=='NOVEL').sum()}/{len(routes)} NOVEL candidates")
    text += f"The default router ended with {int(reference.at['rae','num_experts'])} expert(s); "+"; ".join(route_counts)+". "
    text += f"Always Expand reached {int(reference.at['always_expand','num_experts'])} experts. These runs exercise real cloned-expert training. None of these pilot results establishes superiority over full-backbone replay.\n\n"
    text += "At domain arrival, novelty and prediction error have a positive association. This association is pooled over domains and may reflect chemistry, target shifts or sample complexity. It does not establish that the novelty thresholds are calibrated to property-error risk.\n\n"
    text += "The joint oracle is designed to train once using all domain training/validation partitions; it was interrupted before final evaluation and is omitted from the comparison. All completed methods use current-domain training and bounded retained replay only. Random and structural replay use extra optimizer steps for replay examples; the comparison is not equal-compute. Structural replay uses one additional frozen stage-1 encoder for selection, reported as auxiliary training parameters.\n\n"
    text += "This report covers the 5000-sample pilot only. The repository also provides 30000/full-data profiles and three full-run seeds, without completed large-scale results. No test-driven router hyperparameter search was performed.\n"
    (out/"RESULTS.md").write_text(text)
    print(table.to_string(index=False),flush=True)
    print("REPORT_WRITTEN",out,flush=True)


if __name__ == "__main__":
    main()
