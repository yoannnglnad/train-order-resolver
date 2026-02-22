import polars as pl
import os

# --- Configuration ---
RAW_DATA_PATH = "data/raw"
OUTPUT_DATA_PATH = "data/dataset"

# Entrée : Les horaires (GTFS)
STOP_TIMES_FILE = os.path.join(RAW_DATA_PATH, "stop_times.txt")

# Sortie : Les connexions pondérées par le temps
OUTPUT_FILE = os.path.join(OUTPUT_DATA_PATH, "connections.parquet")

def str_time_to_minutes(col_name):
    """
    Convertit une chaîne 'HH:MM:SS' en minutes (Float).
    Gère les heures GTFS > 24h (ex: '25:10:00' devient 1510 minutes).
    """
    # On découpe la chaine par ':'
    # index 0 = Heures, index 1 = Minutes, index 2 = Secondes
    parts = pl.col(col_name).str.split(":")
    
    hours = parts.list.get(0).cast(pl.Int32)
    minutes = parts.list.get(1).cast(pl.Int32)
    seconds = parts.list.get(2).cast(pl.Int32)
    
    return (hours * 60) + minutes + (seconds / 60)

def main():
    print("⏳ Démarrage : Construction du Graphe basé sur le TEMPS de trajet...")

    # Lecture Lazy du fichier (mémoire efficace)
    # stop_times contient: trip_id, arrival_time, departure_time, stop_id, stop_sequence
    q = (
        pl.scan_csv(STOP_TIMES_FILE, dtypes={'stop_sequence': pl.Int32})
        .select(['trip_id', 'stop_id', 'departure_time', 'arrival_time', 'stop_sequence'])
        
        # 1. Nettoyage des IDs (StopPoint:OCE87... -> 87...)
        .with_columns(
            pl.col("stop_id").str.extract(r'(\d+)$', 1).alias("uic")
        )
        # Filtre de sécurité sur les IDs
        .filter(pl.col("uic").is_not_null())
        
        # 2. Conversion des heures en minutes (float)
        # departure_time sert pour le nœud de départ
        # arrival_time servira pour le nœud d'arrivée (via shift)
        .with_columns([
            str_time_to_minutes("departure_time").alias("dep_min"),
            str_time_to_minutes("arrival_time").alias("arr_min")
        ])
        
        # 3. Tri pour aligner les stations
        .sort(['trip_id', 'stop_sequence'])
        
        # 4. Décalage pour créer la liaison Gare A -> Gare B
        .with_columns([
            pl.col("uic").shift(-1).alias("to_id"),
            pl.col("trip_id").shift(-1).alias("next_trip_id"),
            # On récupère l'heure d'arrivée à la station B
            pl.col("arr_min").shift(-1).alias("next_arr_min")
        ])
        
        # 5. Calcul du temps de trajet (Weight) et filtrage
        .with_columns(
            (pl.col("next_arr_min") - pl.col("dep_min")).alias("weight")
        )
        .filter(
            (pl.col("trip_id") == pl.col("next_trip_id")) &  # Même train
            (pl.col("to_id").is_not_null()) &
            (pl.col("uic") != pl.col("to_id")) &           # Pas de boucle sur soi-même
            (pl.col("weight") > 0)                         # Temps positif uniquement
        )
        
        # 6. Renommage et Normalisation
        .select([
            pl.col("uic").alias("from_id"),
            "to_id",
            "weight"
        ])
    )

    print("🔄 Agrégation des connexions...")
    # Sur une année, le trajet Paris-Lyon existe 5000 fois.
    # Nous devons réduire cela à une seule arête dans le graphe.
    
    unique_connections = (
        q
        # Normalisation non-orientée (A->B devient min(A,B)->max(A,B))
        .select([
            pl.min_horizontal(["from_id", "to_id"]).alias("start"),
            pl.max_horizontal(["from_id", "to_id"]).alias("end"),
            "weight"
        ])
        # Agrégation :
        # - mean : temps moyen (lisse les retards et les express/omnibus)
        # - median : souvent plus robuste (évite d'être faussé par un TGV ultra lent exceptionnel)
        # - min : le temps RECORD possible sur cette ligne.
        .group_by(["start", "end"])
        .agg(
            pl.col("weight").median().round(2).alias("weight")
        )
        # On remet les noms corrects
        .select([
            pl.col("start").alias("from_id"),
            pl.col("end").alias("to_id"),
            pl.col("weight")
        ])
        .collect()
    )

    print(f"💾 Sauvegarde de {len(unique_connections)} arcs pondérés (minutes)...")
    
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)
    unique_connections.write_parquet(OUTPUT_FILE)

    print("✅ Terminé !")
    print("\n--- Exemple de données (weight = minutes) ---")
    print(unique_connections.head(5))

if __name__ == "__main__":
    main()