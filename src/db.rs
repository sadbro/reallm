use std::future::{IntoFuture, ready, Ready};
use std::iter::zip;
use std::ops::Not;
use qdrant_client::{Payload, Qdrant};
use qdrant_client::qdrant::{CreateCollection, Distance, PointStruct, UpsertPoints, VectorParams, VectorsConfig};
use qdrant_client::qdrant::vectors_config::Config;
use rust_bert::pipelines::sentence_embeddings::Embedding;
use serde_json::json;

pub(crate) const QDRANT_01_EP: &str = "https://33527441-bf34-46f7-8b2c-2cb28d73f690.us-east4-0.gcp.cloud.qdrant.io:6334";
pub(crate) const QDRANT_API_KEY: &str = "f42bNhvABNR5SdiXLsGRQwRmWfMSXu-2eTtCtLlmngduYCFes99v6g";

pub(crate) struct DataPoint {
    pub collection_name: String,
    pub client: Qdrant,
    pub segments: Vec<String>,
    pub embeddings: Vec<Embedding>
}

impl DataPoint {
    pub(crate) fn new(collection_name: String, client: Qdrant, segments: Vec<String>, embeddings: Vec<Embedding>) -> Self {
        DataPoint {
            collection_name,
            client,
            segments,
            embeddings
        }
    }
}

impl IntoFuture for DataPoint {
    type Output = DataPoint;
    type IntoFuture = Ready<Self::Output>;
    fn into_future(self) -> Self::IntoFuture {
        ready(self)
    }
}

const fn get_lazy_collection(collection_name: String, vectors_config: Option<VectorsConfig>) -> CreateCollection {
    CreateCollection {
        collection_name,
        hnsw_config: None,
        wal_config: None,
        optimizers_config: None,
        shard_number: None,
        on_disk_payload: None,
        timeout: None,
        vectors_config,
        replication_factor: None,
        write_consistency_factor: None,
        init_from_collection: None,
        quantization_config: None,
        sharding_method: None,
        sparse_vectors_config: None,
        strict_mode_config: None,
    }
}

// distance below ->
// #[repr(i32)]
// pub enum Distance {
//     UnknownDistance = 0,
//     Cosine = 1,
//     Euclid = 2,
//     Dot = 3,
//     Manhattan = 4,
// }

const fn get_lazy_config(size: u64, distance: i32) -> Config {
    Config::Params(
        VectorParams {
            size,
            distance,
            hnsw_config: None,
            quantization_config: None,
            on_disk: None,
            datatype: None,
            multivector_config: None,
        }
    )
}

fn get_lazy_points(segments: Vec<&str>, embeddings: Vec<Embedding>) -> Vec<PointStruct> {
    let mut points = Vec::new();
    let mut count = 1;
    for (segment, embedding) in zip(segments, embeddings) {
        points.push(
            PointStruct::new(
                count,
                embedding,
                Payload::try_from(
                    json!({
                        "context" : segment
                    })
                ).unwrap()
            )
        );
        count += 1;
    }
    return points;
}

pub(crate) fn connect(endpoint: &'static str, api_key: &'static str, error_message: &'static str) -> Qdrant {
    let client = Qdrant::from_url(endpoint).api_key(api_key).build().expect(error_message);
    println!("DB Connected...");
    return client;
}

pub(crate) async fn print_collections(client: &Qdrant, error_message: &'static str) {
    let collections = client.list_collections().await.expect(error_message);
    println!("query: GET All Collections");
    println!("Found {} collection(s)...", collections.collections.len());
    println!("Time taken: {} us", collections.time * 1000.0);
    for collection in &collections.collections {
        println!("Name: {}", collection.name);
    }
}

pub(crate) async fn setup_collection(client: &Qdrant, collection_name: &str, vector_size: u64, vector_distance: Distance, error_message: &'static str) {
    let found_existence = client.collection_exists(collection_name).await.expect(error_message);
    println!("query: CREATE Collection if not existing");
    if found_existence {
        println!("Collection '{}' already exists, skipping build...", collection_name);
    } else {
        println!("Collection '{}' not found, building collection...", collection_name);
        let created = client.create_collection(
            get_lazy_collection(
                collection_name.to_string(),
                Option::from(
                    VectorsConfig {
                        config: Option::from(
                            get_lazy_config(
                                vector_size,
                                vector_distance.into()
                            )
                        ),
                    }
                )
            )
        ).await.expect(error_message);
        println!("Time taken: {} us", created.time * 1000.0);
        println!("Build status: {}", if created.result {"SUCCESSFUL"} else {"FAILED"});
    }
}

pub(crate) async fn compile_collection_from_document(client: &Qdrant, collection_name: &str, segments: Vec<&str>, embeddings: Vec<Embedding>, error_message: &'static str) {
    println!("query: Compile Collection from document");
    let found_existence = client.collection_exists(collection_name).await.expect(error_message);
    if found_existence.not() {
        println!("Collection does not exist!!!");
    } else {
        let compiled = client.upsert_points(
            UpsertPoints { 
                collection_name: collection_name.to_string(),
                wait: None,
                points: get_lazy_points(segments, embeddings),
                ordering: None,
                shard_key_selector: None
            }
        ).await.expect(error_message);
        println!("Time taken: {} us", compiled.time * 1000.0);
        println!("Build result: {}", compiled.result.unwrap().status);
    }
}
