import streamlit as st
import pandas as pd
import numpy as np

from src.dataloader import load_dataset
from src.environment import (
    true_eval,
    labels_to_onehot_group,
    NUM_OPTIONS,
    configure_environment,
)
from src.models import train_surrogate_model
from src.algorithms import (
    mofa_surrogate_pareto,
    q_learning_refine,
    nsga2_optimize,
    mopso_optimize,
)

from src.visualization import (
    plot_load_balancing_checkpoints,
    plot_radar_comparison,
    plot_3d_sets,
    plot_2d_projections,
    plot_boxplots,
    plot_comparative_load_balancing,
    get_summary_table,
    plot_hypervolume_comparison,
    plot_convergence_metrics,
    plot_hypervolume_bar,
    plot_3d_comparison_fronts,
)

st.set_page_config(page_title="Optimisation Offloading", layout="wide")
st.title("Démo Optimisation Multi-Objectifs (MOO) - Smart Offloading Optimizer : Système Hybride MOFA–Apprentissage par Renforcement")
st.markdown(" Groupe 4 ING3 IA B")
st.markdown("### Scénario : Offloading de tâches vidéo avec contraintes temps réel")

# --- BARRE LATÉRALE ---
st.sidebar.header("1. Configuration des Données")
data_source = st.sidebar.radio("Source du Dataset", ["Génération Vidéo (Frames)", "Fichier JSON Existant"])

if data_source == "Génération Vidéo (Frames)":
    num_frames = st.sidebar.select_slider("Nombre de Frames", options=[500, 1000, 5000, 10000], value=500)
    s_type = "video"
    json_path = None
else:
    num_frames = 500
    s_type = "json"
    json_path = "data/dataset.json"

st.sidebar.header("2. Configuration Algorithme")
strategy = st.sidebar.selectbox("Stratégie d'Initialisation", ["Aléatoire + Greedy", "Auto-Encodeur Génératif"])
use_active_learning = st.sidebar.checkbox("Activer Active Learning (Dynamic Surrogate + MOFA)", value=True)
use_qlearning = st.sidebar.checkbox("Activer Raffinement Q-Learning", value=True)
iterations = st.sidebar.slider("Itérations MOFA", 10, 200, 60)

st.sidebar.markdown("---")
st.sidebar.header("3. Comparaison")
run_benchmarks = st.sidebar.checkbox("Comparer avec NSGA-II + MOPSO", value=False)

# --- EXECUTION ---
if st.sidebar.button("Lancer l'Optimisation"):
    with st.spinner("Chargement..."):
        tasks, groups, num_groups, meta = load_dataset(
            source_type=s_type,
            num_frames=num_frames,
            json_path=json_path,
            seed=42,
        )

        # CRITIQUE: en cas d'utilisation du dataset JSON, reconfigure l’environnement pour coller au dataset
        if s_type == "json":
            configure_environment(
                node_vms=meta.get("node_profiles"),
                global_constants=meta.get("global_constants"),
                group_size=meta.get("group_size"),
            )

        st.success(f"Données : {len(tasks)} tâches / {num_groups} groupes.")

    with st.spinner("Entraînement IA..."):
        N_TRAIN = 300
        rng = np.random.default_rng(42)

        X_train, Y_train = [], []
        for _ in range(N_TRAIN):
            labels = rng.integers(0, NUM_OPTIONS, size=num_groups)
            x_flat = labels_to_onehot_group(labels, num_groups)
            X_train.append(x_flat)
            Y_train.append(true_eval(x_flat, tasks, groups, num_groups))

        X_train = np.array(X_train, dtype=np.float32)
        Y_train = np.array(Y_train, dtype=np.float32)

        ae, mlp, Y_min, Y_rng = train_surrogate_model(
            X_train,
            Y_train,
            X_train.shape[1],
            epochs_ae=20,
            epochs_sur=30,
        )

    with st.spinner("Optimisation MOFA..."):
        use_ae = (strategy == "Auto-Encodeur Génératif")
        mofa_X, mofa_F, history = mofa_surrogate_pareto(
            tasks,
            groups,
            num_groups,
            ae,
            mlp,
            Y_min,
            Y_rng,
            n_iter=iterations,
            use_ae_init=use_ae,
            active_learning=bool(use_active_learning),
        )

    final_X, final_F = mofa_X, mofa_F
    if use_qlearning:
        with st.spinner("Q-Learning..."):
            final_X, final_F = q_learning_refine(
                mofa_X,
                mofa_F,
                tasks,
                groups,
                num_groups,
                episodes=30,
            )

    # Baseline (sécurisé)
    baseline_F = Y_train[:100] if len(Y_train) >= 100 else Y_train

    # Benchmarks
    nsga2_F, mopso_F = [], []
    nsga2_X, mopso_X = [], []
    hist_nsga, hist_mopso = {}, {}

    if run_benchmarks:
        with st.spinner("Exécution Benchmarks (NSGA-II + MOPSO)..."):
            nsga2_X, nsga2_F, hist_nsga = nsga2_optimize(tasks, groups, num_groups, n_iter=iterations)
            mopso_X, mopso_F, hist_mopso = mopso_optimize(tasks, groups, num_groups, n_iter=iterations)

    # --- VISUALISATION ---
    st.divider()
    tab1, tab2 = st.tabs(["💻 Résultats MOFA", "📈 Comparaison Algorithmes"])

    with tab1:
        st.markdown("### Analyse de la Convergence MOFA")

        st.markdown("**Dynamique d'Apprentissage (Qualité vs Diversité)**")
        st.pyplot(plot_convergence_metrics(history), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Progression de l'Hypervolume**")
            st.pyplot(plot_hypervolume_bar(history), use_container_width=True)

        with col2:
            st.markdown("**Stratégie de Load Balancing**")
            st.pyplot(plot_load_balancing_checkpoints(history), use_container_width=True)

        st.divider()
        c3, c4 = st.columns(2)
        with c3:
            st.markdown("**Front de Pareto Final (3D)**")
            st.pyplot(plot_3d_sets(baseline_F, final_F), use_container_width=True)
        with c4:
            st.markdown("**Compromis (Radar)**")
            st.pyplot(plot_radar_comparison(baseline_F, final_F), use_container_width=True)

    with tab2:
        if run_benchmarks:
            st.markdown("#### 0. Comparaison 3D (Baseline + Fronts)")
            fronts = {
                "MOFA": mofa_F,
                "MOFA+RL": final_F,
                "NSGA-II": nsga2_F,
                "MOPSO": mopso_F,
            }
            st.pyplot(
                plot_3d_comparison_fronts(
                    baseline_F,
                    fronts,
                    title="Nuage baseline + fronts Pareto (MOFA / MOFA+RL / NSGA-II / MOPSO)",
                ),
                use_container_width=True,
            )

            methods_F = {
                "Baseline": baseline_F,
                "MOFA+RL": final_F,
                "NSGA-II": nsga2_F,
                "MOPSO": mopso_F,
            }
            methods_X = {
                "MOFA+RL": final_X,
                "NSGA-II": nsga2_X,
                "MOPSO": mopso_X,
            }
            histories_hv = {
                "MOFA": history,
                "NSGA-II": hist_nsga,
                "MOPSO": hist_mopso,
            }

            st.markdown("#### 1. Convergence (Hypervolume)")
            st.pyplot(plot_hypervolume_comparison(histories_hv), use_container_width=True)

            st.markdown("#### 2. Projections 2D des Objectifs")
            st.pyplot(plot_2d_projections(methods_F), use_container_width=True)

            st.markdown("#### 3. Distributions (Boxplots)")
            st.pyplot(plot_boxplots(methods_F), use_container_width=True)

            st.markdown("#### 4. Comparaison Load Balancing Moyen")
            st.pyplot(plot_comparative_load_balancing(methods_X, num_groups), use_container_width=True)

            st.markdown("#### 5. Tableau Résumé")
            df = get_summary_table(methods_F)
            if not df.empty:
                st.dataframe(df.style.highlight_min(subset=["Latence (Moy)", "Coût (Moy)"], color="lightgreen"))
            else:
                st.warning("Tableau résumé vide (aucune donnée exploitable).")
        else:
            st.warning("Activez la case 'Comparer avec NSGA-II + MOPSO' dans la barre latérale pour voir ces graphiques.")
else:
    st.info("👈 Configurez et lancez.")
