import os
import pandas as pd
import logging
import argparse

logger = logging.getLogger(__name__)
logging.basicConfig(filename="load_data.log")


tracelengths = { "airportA-3-50-30" : [100,500],
                 "airportB-3-50-30" : [100,500],
                 "airportA-7-50-30": [100, 500],
                 "airportB-7-50-30": [100, 500],
                 "evadeI-10" : [100,500],
                 "evadeI-15": [100, 500],
                 "refuelA-12-50": [100, 500],
                 "refuelB-12-50": [100, 500],
                 "evadeV-5-3" : [100,500],
                 "evadeV-6-3": [100, 500]
                 }

experiments = [ "airportA-7-50-30","airportB-3-50-30", "airportB-7-50-30",
                 "refuelA-12-50", "refuelB-12-50","evadeI-15", "evadeV-5-3", "evadeV-6-3"]


parser = argparse.ArgumentParser(description='Process stats.')
parser.add_argument('folder', type=str,
                   help='stats from which folder to use')

args = parser.parse_args()
stats_folder = args.folder


method_based_online_stats = {
    "unf" : [("McTime", "mc_time"), ("UnfTime", "unf_time"), ("TotalTime", "totaltime"), ("MdpStates", "mdp_states")],
    "ff" : [("TrackTime", "track_time"), ("ReduceTime", "red_time"), ("TotalTime", "totaltime"),("NrBeliefsBR", "origcard") , ("NrBeliefsAR", "redcard") , ("Dimension", "dim")] ,
}

def compute_online_statistics(folder, TPs, method):
    tables = {}
    if not os.path.isdir(folder):
        print(f"Warning! {folder} not found.")
        return {}
    for filename in os.listdir(folder):
        if filename.endswith(".out"):
            continue
        with open(os.path.join(folder, filename), 'r') as f:
            tables[filename] = pd.read_csv(f)


    tp_tables = {}
    tp_statistics = {}
    for tp in TPs:
        filtered_tables = []
        for filename, table in tables.items():
            if tp - 1 not in table.index:
                logger.warning(f"Time out for tp={tp} in file {filename}")
                continue
            if 1.0 in table.loc[table['Index'] == tp - 1].TimedOut:
                logger.warning(f"Time out for tp={tp} in file {filename}")
                continue
            filtered_tables.append(table)
        tp_statistics[tp] = { "timedOut": len(tables) - len(filtered_tables) ,"passed": len(filtered_tables)}

        if len(filtered_tables) > 0:
            tp_tables[tp] = pd.concat([table[table['Index'] == tp - 1] for table in filtered_tables])
            for stat in method_based_online_stats[method]:
                column_name, id = stat[0], stat[1]
                tp_statistics[tp][f"{id}_min"] = tp_tables[tp][column_name].min()
                tp_statistics[tp][f"{id}_p10"] = tp_tables[tp][column_name].quantile(0.1)
                tp_statistics[tp][f"{id}_avg"] = tp_tables[tp][column_name].mean()
                tp_statistics[tp][f"{id}_p90"] = tp_tables[tp][column_name].quantile(0.9)
                tp_statistics[tp][f"{id}_max"] = tp_tables[tp][column_name].max()
    return tp_statistics



def compute_offline_statistics(folder, TPs, method):
    tables = {}
    if not os.path.isdir(folder):
        print(f"Warning! {folder} not found.")
        return {}
    for filename in os.listdir(folder):
        if filename.endswith(".out"):
            continue
        with open(os.path.join(folder, filename), 'r') as f:
            tables[filename] = pd.read_csv(f)

    tp_tables = {}
    tp_statistics = {}
    for tp in TPs:
        filtered_tables = []
        for filename, table in tables.items():
            if tp - 1 not in table.index:
                logger.warning(f"Time out for tp={tp} in file {filename}")
                continue
            if 1.0 in table.loc[table['Index'] == tp - 1].TimedOut:
                logger.warning(f"Time out for tp={tp} in file {filename}")
                continue
            filtered_tables.append(table)
        tp_statistics[tp] = { "timedOut": len(tables) - len(filtered_tables),"passed": len(filtered_tables)}

        if len(filtered_tables) > 0:
            tp_tables[tp] = pd.concat([table.loc[table['Index'] <= tp - 1].sum(numeric_only=True).to_frame() for table in filtered_tables], axis=1).transpose()
            if method == "unf":
                tp_tables[tp]["UnfFrac"] = tp_tables[tp].apply(lambda row: row.UnfTime / row.TotalTime, axis=1)
            if method == "ff":
                tp_tables[tp]["BElim"] = tp_tables[tp].apply(lambda row: row.NrBeliefsBR - row.NrBeliefsAR, axis=1)
            #print(tp_tables[tp])
            for stat in method_based_offline_stats[method]:
                column_name, id = stat[0], stat[1]
                tp_statistics[tp][f"{id}_min"] = tp_tables[tp][column_name].min()
                tp_statistics[tp][f"{id}_p10"] = tp_tables[tp][column_name].quantile(0.1)
                tp_statistics[tp][f"{id}_avg"] = tp_tables[tp][column_name].mean()
                tp_statistics[tp][f"{id}_p90"] = tp_tables[tp][column_name].quantile(0.9)
                tp_statistics[tp][f"{id}_max"] = tp_tables[tp][column_name].max()
    return tp_statistics



def export_online_stats():
    tablehead = r"""
     %auto generated 
     \begin{tabular}{lll|rrr|rrrrrrr|rrrrr|}
    &      &      &          &               &         & \multicolumn{7}{c|}{Forward Filtering} & \multicolumn{5}{c|}{Unrolling}  \\
    Id & Name & Inst & $|S|$ & $|P|$ & $|\trace|$ & $N$ &  $\underset{\text{avg}}{T}$ & $\underset{\text{max}}{T}$ &  $\underset{\text{avg}}{B}$ & $\underset{\text{max}}{B}$     & $\underset{\text{avg}}{D}$ & $\underset{\text{max}}{D}$ & $N$ &  $\underset{\text{avg}}{T}$ & $\underset{\text{max}}{T}$ & $\underset{\text{avg}}{|S_u|}$ & $\underset{\text{max}}{|S_u|}$ \\\hline
    """
    tablefooter = r"""
    \end{tabular}
    """
    with open("table1.tex", 'w') as texfile:
        texfile.write(tablehead + "\n")
        for idx, benchmark in enumerate(experiments):
            nr_states = "not-found"
            nr_transitions = "not-found"
            try:
                with open (f"{stats_folder}/{benchmark}-unf-ea/stats.out") as statfile:
                    for line in statfile:
                        if line.startswith("//"):
                            continue
                        kv = line.strip().split("=")
                        if kv[0] == "states":
                            nr_states = kv[1]
                        if kv[0] == "transitions":
                            nr_transitions = kv[1]
            except:
                print(f"Warning. (Likely: File {stats_folder}/{benchmark}-unf-ea/stats.out not found.)")
            data = benchmark.split("-")
            texfile.write("\multirow{2}{*}{")
            texfile.write(str(idx+1))
            texfile.write("}\t& \multirow{2}{*}{\\benchmark{")
            texfile.write(data[0])
            texfile.write("}}\t& \multirow{2}{*}{")
            texfile.write(",".join(data[1:]))
            texfile.write("}\t& \multirow{2}{*}{")
            texfile.write(nr_states)
            texfile.write("}\t& \multirow{2}{*}{")
            texfile.write(nr_transitions)
            ffstats = compute_online_statistics(f"{stats_folder}/{benchmark}-ff-ch-ea", tracelengths[benchmark], "ff")
            unfstats = compute_online_statistics(f"{stats_folder}/{benchmark}-unf-ea", tracelengths[benchmark], "unf")
            texfile.write("} & ")
            for tl in tracelengths[benchmark]:
                if tl != tracelengths[benchmark][0]:
                    texfile.write("\t\t\t&\t&\t&\t&\t&")
                texfile.write(str(tl))
                texfile.write("\t&")
                if len(ffstats) == 0:
                    texfile.write("\t&")
                else:
                    if ffstats[tl]["timedOut"] > 0:
                        texfile.write("\\highlightFAIL{")
                    else:
                        assert ffstats[tl]["timedOut"] == 0
                        texfile.write("\\highlightPASS{")
                    texfile.write("{}".format(ffstats[tl]["passed"]))
                    texfile.write("}\t& ")
                if len(ffstats) > 0 and ffstats[tl]["passed"] > 0:
                    texfile.write("{:.2f}".format(ffstats[tl]["totaltime_avg"]))
                    texfile.write("\t& ")
                    texfile.write("{:.2f}".format(ffstats[tl]["totaltime_max"]))
                    texfile.write("\t& ")
                    texfile.write("{:.1f}".format(ffstats[tl]["redcard_avg"]))
                    texfile.write("\t& ")
                    texfile.write("{:.0f}".format(ffstats[tl]["redcard_max"]))
                    texfile.write("\t& ")
                    texfile.write("{:.1f}".format(ffstats[tl]["dim_avg"]))
                    texfile.write("\t& ")
                    texfile.write("{:.0f}".format(ffstats[tl]["dim_max"]))
                else:
                    texfile.write("\t& ")
                    texfile.write("\t& ")
                    texfile.write("\t& ")
                    texfile.write("\t& ")
                    texfile.write("\t& ")
                texfile.write("\t& ")
                if len(unfstats) == 0:
                    texfile.write("\t&")
                else:
                    if unfstats[tl]["timedOut"] > 0:
                        texfile.write("\\highlightFAIL{")
                    else:
                        assert unfstats[tl]["timedOut"] == 0
                        texfile.write("\\highlightPASS{")
                    texfile.write("{}".format(unfstats[tl]["passed"]))
                    texfile.write("}\t& ")

                if len(unfstats) > 0 and unfstats[tl]["passed"] > 0:
                    texfile.write("{:.2f}".format(unfstats[tl]["totaltime_avg"]))
                    texfile.write("\t& ")
                    texfile.write("{:.2f}".format(unfstats[tl]["totaltime_max"]))
                    texfile.write("\t& ")
                    texfile.write("{:.0f}".format(unfstats[tl]["mdp_states_avg"]))
                    texfile.write("\t& ")
                    texfile.write("{:.0f}".format(unfstats[tl]["mdp_states_max"]))
                else:
                    texfile.write("\t& ")
                    texfile.write("\t& ")
                    texfile.write("\t& ")
                texfile.write("\\\\\n")

            texfile.write("\\hline\n")
        texfile.write(tablefooter)

def export_totals_table():
    with open("table2.tex", 'w') as texfile:
        tableheader = r"""
          \begin{tabular}{lr|rrrrr|rrrr|rrrrr|rrr|}
             &      &  \multicolumn{5}{c|}{FF w/ CH}  &  \multicolumn{4}{c|}{FF w/o CH}      & \multicolumn{5}{c|}{UNR (exact)} & \multicolumn{3}{c|}{UNR (ovi)}   \\
        Id & $|\trace|$ & $N$ & $\underset{\text{avg}}{T}$ & $\underset{\text{max}}{T}$ & $\underset{\text{avg}}{B}$ & $\underset{\text{avg}}{E}$ & $N$ & $\underset{\text{avg}}{T}$ & $\underset{\text{max}}{T}$ & $\underset{\text{avg}}{B}$  & $N$ & $\underset{\text{avg}}{T}$ & $\underset{\text{max}}{T}$ &    $\underset{\text{avg}}{Bld^\%}$ & $\underset{\text{max}}{Bld^\%}$ & N  & $\underset{\text{avg}}{T}$ & $\underset{\text{max}}{T}$  \\\hline
        """
        tablefooter = r"""
        \end{tabular}
        """
        texfile.write(tableheader + "\n")
        for idx, benchmark in enumerate(experiments):
            data = benchmark.split("-")
            texfile.write(f"% {benchmark}\n")
            texfile.write(str(idx+1))
            texfile.write("\t& ")
            ff_nr_stats = compute_offline_statistics(f"{stats_folder}/{benchmark}-ff-nr-ea", tracelengths[benchmark], "ff")
            ff_ch_stats = compute_offline_statistics(f"{stats_folder}/{benchmark}-ff-ch-ea", tracelengths[benchmark], "ff")
            unf_ea_stats = compute_offline_statistics(f"{stats_folder}/{benchmark}-unf-ea", tracelengths[benchmark], "unf")
            unf_fl_stats = compute_offline_statistics(f"{stats_folder}/{benchmark}-unf-fl", tracelengths[benchmark], "unf")
            # texfile.write(" & ")
            for tl in tracelengths[benchmark]:
                 if tl != tracelengths[benchmark][0]:
                     texfile.write("\t\t\t&\t")
                 texfile.write(str(tl))
                 texfile.write("\t& ")
                 if len(ff_ch_stats) == 0:
                     texfile.write("\t& ")
                 else:
                     if ff_ch_stats[tl]["timedOut"] > 0:
                         texfile.write("\\highlightFAIL{")
                     else:
                         assert ff_ch_stats[tl]["timedOut"] == 0
                         texfile.write("\\highlightPASS{")
                     texfile.write("{}".format(ff_ch_stats[tl]["passed"]))
                     texfile.write("}\t& ")
                 if len(ff_ch_stats) > 0 and ff_ch_stats[tl]["passed"] > 0:
                     texfile.write("{:.1f}".format(ff_ch_stats[tl]["totaltime_avg"]))
                     texfile.write("\t& ")
                     texfile.write("{:.1f}".format(ff_ch_stats[tl]["totaltime_max"]))
                     texfile.write("\t& ")
                     texfile.write("{:.0f}".format(ff_ch_stats[tl]["redcard_avg"]))
                     texfile.write("\t& ")
                     texfile.write("{:.0f}".format(ff_ch_stats[tl]["elimcard_avg"]))
                 else:
                     texfile.write("\t& ")
                     texfile.write("\t& ")
                     texfile.write("\t& ")
                 texfile.write("\t& ")
                 if len(ff_nr_stats) == 0:
                     texfile.write("\t& ")
                 else:
                     if ff_nr_stats[tl]["timedOut"] > 0:
                         texfile.write("\\highlightFAIL{")
                     else:
                         assert ff_nr_stats[tl]["timedOut"] == 0
                         texfile.write("\\highlightPASS{")
                     texfile.write("{}".format(ff_nr_stats[tl]["passed"]))
                     texfile.write("}\t& ")
                 if len(ff_nr_stats) > 0 and ff_nr_stats[tl]["passed"] > 0:
                     texfile.write("{:.1f}".format(ff_nr_stats[tl]["totaltime_avg"]))
                     texfile.write("\t& ")
                     texfile.write("{:.1f}".format(ff_nr_stats[tl]["totaltime_max"]))
                     texfile.write("\t& ")
                     texfile.write("{:.0f}".format(ff_nr_stats[tl]["redcard_avg"]))
                 else:
                     texfile.write("\t& ")
                     texfile.write("\t& ")
                 texfile.write("\t& ")
                 if len(unf_ea_stats) == 0:
                     texfile.write("\t& ")
                 else:
                     if unf_ea_stats[tl]["timedOut"] > 0:
                         texfile.write("\\highlightFAIL{")
                     else:
                         assert unf_ea_stats[tl]["timedOut"] == 0
                         texfile.write("\\highlightPASS{")
                     texfile.write("{}".format(unf_ea_stats[tl]["passed"]))
                     texfile.write("}\t& ")
                 if len(unf_ea_stats) > 0 and unf_ea_stats[tl]["passed"] > 0:
                     texfile.write("{:.1f}".format(unf_ea_stats[tl]["totaltime_avg"]))
                     texfile.write("\t& ")
                     texfile.write("{:.1f}".format(unf_ea_stats[tl]["totaltime_max"]))
                     texfile.write("\t& ")
                     texfile.write("{:.0f}".format(100*unf_ea_stats[tl]["unffrac_avg"]))
                     texfile.write("\t& ")
                     texfile.write("{:.0f}".format(100*unf_ea_stats[tl]["unffrac_max"]))
                     texfile.write("\t& ")
                 else:
                     texfile.write("\t& ")
                     texfile.write("\t& ")
                     texfile.write("\t& ")
                     texfile.write("\t& ")
                 if len(unf_fl_stats) > 0:
                     if unf_fl_stats[tl]["timedOut"] > 0:
                         texfile.write("\\highlightFAIL{")
                     else:
                         assert unf_fl_stats[tl]["timedOut"] == 0
                         texfile.write("\\highlightPASS{")
                     texfile.write("{}".format(unf_fl_stats[tl]["passed"]))
                     texfile.write("}\t& ")
                 else:
                     texfile.write("\t& ")
                 if len(unf_fl_stats) > 0 and unf_fl_stats[tl]["passed"] > 0:
                     texfile.write("{:.1f}".format(unf_fl_stats[tl]["totaltime_avg"]))
                     texfile.write("\t& ")
                     texfile.write("{:.1f}".format(unf_fl_stats[tl]["totaltime_max"]))
                 else:
                     texfile.write("\t& ")
                 texfile.write("\\\\\n")
            texfile.write("\\hline\n")
        texfile.write(tablefooter)

export_online_stats()
export_totals_table()