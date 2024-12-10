mod db;

use qdrant_client::qdrant::Distance;
use rust_bert::pipelines::sentence_embeddings::{SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType};
use crate::db::{compile_collection_from_document, DataPoint, print_collections, setup_collection};

async fn setup() -> DataPoint {
    let collection_name = "test_col_mini_lm_l12_v2".to_owned();
    let client = db::connect(db::QDRANT_01_EP, db::QDRANT_API_KEY, "Error connecting to DB");

    let content = tokio::fs::read_to_string("src/subjects/text-qa.txt")
        .await
        .expect("Error reading file");

    let segments: Vec<String> = content.lines().map(String::from).collect();

    // Clone segments for the blocking task to prevent moving the original
    let segments_for_embedding = segments.clone();

    // Offload the embedding model setup and embedding to a blocking thread
    let embeddings = tokio::task::spawn_blocking(move || {
        let embedding_model = SentenceEmbeddingsBuilder::remote(
            SentenceEmbeddingsModelType::AllMiniLmL12V2
        ).create_model()
            .expect("Creating embedding model failed");

        embedding_model
            .encode(&segments_for_embedding)
            .expect("Embedding failed.")
    }).await.expect("Embedding task failed");

    println!("{} embeddings generated.", embeddings.len());

    DataPoint::new(
        collection_name,
        client,
        segments,   // Original segments are used here
        embeddings,
    )
}

#[tokio::main]
async fn main() {
    // let data_points = setup().await;
    // let segments: Vec<&str> = data_points.segments.iter().map(AsRef::as_ref).collect();
    // 
    // setup_collection(
    //     &data_points.client,
    //     &data_points.collection_name,
    //     384,
    //     Distance::Cosine,
    //     "Error building collection"
    // )
    //     .await;
    // 
    // compile_collection_from_document(
    //     &data_points.client,
    //     &data_points.collection_name,
    //     segments,
    //     data_points.embeddings,
    //     "Error compiling embedding"
    // )
    //     .await;
    let client = db::connect(db::QDRANT_01_EP, db::QDRANT_API_KEY, "Error connecting to DB");
    print_collections(&client, "Error listing all collections").await;
}
