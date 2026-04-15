# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import sys
import warnings

# Suppress specific warning from huggingface_hub
warnings.filterwarnings("ignore", category=FutureWarning, message="`resume_download` is deprecated and will be removed in version 1.0.0.")


import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
# import tensorflow_hub as hub
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import time
import re

import os
import json

from config import FLAGS
from saver import saver
print = saver.log_info

class RAGAgent:
    def __init__(self, documents, rag_type, embedding_model='sbert'):
        """
        Initialize the RAGAgent with a list of documents, an embedding model, and a rag_type.
        
        :param documents: List of document chunks (strings).
        :param embedding_model: The embedding model to use ('sbert', 'use', 'bert').
        :param rag_type: The type of RAG operation ('Suggestions', 'Examples').
        """
        self.delimiter = "<---->"
        self.rag_type = rag_type
        # Extract sections for comparison and retrieval
        # self.documents = documents
        
        self.qtree_engine = None

        # Filter documents based on 'Functionality == 1.0' if needed
        if rag_type == "Suggestions":
            if FLAGS.filter_functionality:
                documents = [doc for doc in documents if self.check_functionality(doc)]
                print(f"@@@DEBUG: Length of the filtered documents is @@@ {len(documents)} @@@")
            
        documents_storage = sum(sys.getsizeof(doc) for doc in documents)
        documents_storage_mb = documents_storage / (1024 * 1024)  # Convert bytes to MB
        # print(f"Storage of documents: {documents_storage_mb:.2f} MB")
        saver.save_stats('documents_storage_mb', documents_storage_mb)


        self.original_documents = [doc.replace(self.delimiter, "\n").strip() for doc in documents]
        self.documents = [self.extract_section(doc, FLAGS.similarity_str, full_string=True) for doc in documents]

        if rag_type == "Suggestions":
            self.retrieve_documents = [self.extract_section(doc, FLAGS.retrieve_str, full_string=True) for doc in documents]
            # breakpoint()
            # print(f"@@@DEBUG: The first retrieved document is @@@ {self.retrieve_documents[0]}")
        elif rag_type == "Examples":
            self.retrieve_documents = [self.extract_section(doc, ["all"], full_string=True) for doc in documents]
            # breakpoint()
            # print(f"@@@DEBUG: The first retrieved document is @@@ {self.retrieve_documents[0]}")

        self.model = self._load_model(embedding_model)
        self.embeddings = self._index_documents()


    def _load_model(self, embedding_model):
        """
        Load the specified embedding model.
        
        :param embedding_model: The embedding model to use ('sbert', 'use', 'bert').
        :return: The loaded model.
        """
        if embedding_model == 'sbert':
            return SentenceTransformer('all-MiniLM-L6-v2')
        elif embedding_model == 'use':
            return hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        elif embedding_model == 'bert':
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = BertModel.from_pretrained('bert-base-uncased')
            return (tokenizer, model)
        else:
            raise ValueError("Unsupported embedding model: choose 'sbert', 'use', or 'bert'")

    def _index_documents(self):
        """
        Index the provided documents using the embedding model and store them in-memory.
        Loads from cache if available, otherwise computes and saves embeddings.
        
        :return: Numpy array of document embeddings.
        """
        # Check if cached embeddings exist
        embeddings_path = os.path.join(FLAGS.load_suggestions_path, 'embeddings.npy')
        
        if os.path.exists(embeddings_path):
            try:
                print(f"Loading cached embeddings from: {embeddings_path}")
                all_embeddings_np = np.load(embeddings_path)
                
                # Verify that the cached embeddings match the current documents
                if all_embeddings_np.shape[0] == len(self.documents):
                    print(f"✓ Loaded {all_embeddings_np.shape[0]} cached embeddings")
                    print("Memory/Storage cost (approx):", all_embeddings_np.nbytes, "bytes")
                    print("Embedding shape:", all_embeddings_np.shape)
                    
                    # Set indexing_runtime to 0 since we loaded from cache
                    saver.save_stats('indexing_runtime', 0.0)
                    return all_embeddings_np
                else:
                    # Shape mismatch - recompute silently (expected when dataset changes)
                    print("Recomputing embeddings...")
            except Exception as e:
                print(f"⚠ Failed to load cached embeddings: {e}")
                print("Recomputing embeddings...")
        
        # Compute embeddings (cache miss or invalid)
        print("Computing embeddings for documents...")
        start_index_time = time.time()

        batch_size = 500  # Adjust batch size as needed
        all_embeddings = []

        for i in tqdm(range(0, len(self.documents), batch_size), desc="Indexing documents"):
            batch_documents = self.documents[i:i+batch_size]
            if isinstance(self.model, SentenceTransformer):
                embeddings = self.model.encode(batch_documents, convert_to_tensor=True)
            elif isinstance(self.model, tuple):
                tokenizer, model = self.model
                tokens = tokenizer(batch_documents, padding=True, truncation=True, return_tensors="pt")
                with torch.no_grad():
                    embeddings = model(**tokens).last_hidden_state.mean(dim=1)
            else:
                embeddings = self.model(batch_documents).numpy()

            embeddings_np = embeddings.cpu().numpy() if hasattr(embeddings, 'cpu') else embeddings
            all_embeddings.append(embeddings_np)

        all_embeddings_np = np.vstack(all_embeddings)

        end_index_time = time.time()
        elapsed_time = self._format_time(end_index_time - start_index_time)

        saver.save_stats('indexing_runtime', end_index_time - start_index_time)
 
        # Print summary statistics
        print("Documents indexed:", len(self.documents))
        print("Memory/Storage cost (approx):", all_embeddings_np.nbytes, "bytes")
        print("Embedding shape:", all_embeddings_np.shape)
        print("Indexing time:", elapsed_time)

        # Save the embeddings to a file
        print(f"Saving embeddings to: {embeddings_path}")
        np.save(embeddings_path, all_embeddings_np)
        
        return all_embeddings_np


    def extract_section(self, doc, section_str_list, full_string=False):
        """
        Extract the part of the document based on the section_str_list.
        
        :param doc: The document string.
        :param section_str_list: The list of sections to extract.
        :param full_string: If True, return the full string instead of extracting sections.
        :return: The extracted sections concatenated with '\n'.
        """
        if "all" in section_str_list:
            return doc.replace(self.delimiter, "\n").strip()
        
        parts = doc.split(self.delimiter)
        doc_dict = {}
        for part in parts:
            key_value = part.split(":", 1)
            if len(key_value) == 2:
                key = key_value[0].strip()
                value = key_value[1].strip()
                doc_dict[key] = value

        if full_string:
            extracted_parts = [f"{key}: {value}" for key, value in doc_dict.items() if key in section_str_list]
        else:
            extracted_parts = [doc_dict.get(section, "") for section in section_str_list]
        
        return "\n".join(extracted_parts).strip()

    def check_functionality(self, doc):
        """
        Check if the document has a "Functionality" score of 1.0.
        
        :param doc: The document string.
        :return: True if the Functionality score is 1.0, otherwise False.
        """
        functionality_str = self.extract_section(doc, ["Functionality"], full_string=False)
        functionality_score = functionality_str.split(":")[1].strip() if ":" in functionality_str else functionality_str.strip()
        return functionality_score == '1.0'

    def retrieve_with_qtree(self, query, top_k=5, testbench=None):
        """
        Retrieve suggestions using Q-Tree based inference.
        Supports multiple modes via FLAGS.qtree_retrieval_mode:
        - 'prompt': Original Q-Tree inference
        - 'initial_sva_generation': Generate SVA, hybrid ranking, operator pruning
        - 'rule_generation': Generate rules from Q-Trees
        
        :param query: The query containing prompt, assertion, etc.
        :param top_k: The number of suggestions to generate.
        :param testbench: Optional testbench code
        :return: List of suggestions based on Q-Tree thinking.
        """
        # Check if enhanced retrieval mode is enabled
        if FLAGS.retrieval_on_ranking:
            # Use enhanced retrieval
            print(f"\n=== Using Ranking-Based Q-Tree Retrieval (Q-Tree Ranking Mode: {FLAGS.qtree_ranking_mode}, Rule Source: {FLAGS.rule_source}) ===")
            
            try:
                from qtree_enhanced_retrieval import integrate_enhanced_retrieval
                
                # Load Q-Tree data from disk
                qtree_file = os.path.join(FLAGS.load_suggestions_path, "qtrees.json")
                if os.path.exists(qtree_file):
                    with open(qtree_file, 'r') as f:
                        qtree_data = json.load(f)
                
                # Get CSV dataframe if available (for training examples)
                csv_df = None
                if hasattr(FLAGS, 'dataset_path'):
                    import pandas as pd
                    csv_df = pd.read_csv(FLAGS.dataset_path)
                
                # Call enhanced retrieval
                result = integrate_enhanced_retrieval(
                    query=query,
                    qtree_data=qtree_data,
                    testbench=testbench,
                    csv_df=csv_df,
                    top_k=top_k
                )

                # print(f"@@@DEBUG: result: {result}")
                # breakpoint()
                
                # Handle new dict return format
                if isinstance(result, dict):
                    suggestions = result.get('suggestions', [])
                    initial_sva = result.get('initial_sva', None)
                    operator_explanations = result.get('operator_explanations', {})
                    print(f"suggestions: {suggestions}")
                    print(f"initial_sva: {initial_sva}")
                    print(f"operator_explanations: {list(operator_explanations.keys())}")
                    
                    # Return suggestions, initial_sva, and operator_explanations so caller can use them
                    # If suggestions is empty, caller can directly use initial_sva
                    # breakpoint()
                    return {'suggestions': suggestions, 'initial_sva': initial_sva, 'operator_explanations': operator_explanations}
                else:
                    # Backward compatibility - if result is a list
                    print(f"suggestions (legacy format): {result}")
                    return result
                
            except Exception as e:
                print(f"Enhanced retrieval failed: {e}")
                print("Falling back to original Q-Tree inference")
                # Fall through to original implementation
        
        # Original Q-Tree inference implementation
        # if not self.qtree_engine:
        #     raise Exception("Q-Tree engine not initialized")
        
        print("\n=== Q-TREE RETRIEVE_WITH_QTREE (Original Mode) ===")
        print("Raw query received:")
        print(query[:200] + "..." if len(query) > 200 else query)
        print("=" * 50)
        
        # Parse the query to extract components
        query_dict = self._parse_query(query)
        for key, value in query_dict.items():
            print(f"  {key}: {value[:100]}..." if len(str(value)) > 100 else f"  {key}: {value}")
        
        prompt = query_dict.get('Question', '').replace('Create a SVA assertion that checks: ', '')
        assertion = query_dict.get('Answer', '')  # May be empty or None
 
        if not prompt:
            print("ERROR: Missing prompt!")
            raise Exception("Could not extract prompt from query")
        
        # Use Q-Tree inference to fix assertion (handles both with/without assertion)
        fixed_assertion, inference_info = self.qtree_engine.fix_assertion_with_qtree(
            prompt=prompt,
            assertion=assertion if assertion else None,
            testbench=testbench if testbench else None
        )
        
        # Generate suggestions based on Q-Tree analysis
        suggestions = []
        
        # Add applied fixes as suggestions
        for fix in inference_info.get('applied_fixes', []):
            if fix['confidence'] >= getattr(FLAGS, 'qtree_decision_confidence_threshold', 0.7):
                suggestions.append(f"- {fix['action']}")
        
        # Add insights from similar Q-Trees
        for task_id, similarity in inference_info.get('similar_qtrees', [])[:3]:
            if task_id in self.qtree_engine.qtree_database:
                qtree = self.qtree_engine.qtree_database[task_id]
                for rule in qtree.get('rules', [])[:2]:  # Top 2 rules from each
                    if rule not in suggestions:
                        suggestions.append(f"- {rule}")
        
        # If not enough suggestions, add general patterns
        if len(suggestions) < top_k:
            for pattern in self.qtree_engine.pattern_database.values():
                for fix in pattern.common_fixes:
                    if f"- {fix}" not in suggestions:
                        suggestions.append(f"- {fix}")
                    if len(suggestions) >= top_k:
                        break
        
        return suggestions[:top_k]
    
    def _parse_query(self, query):
        """Parse query string into dictionary"""
        parts = query.split(self.delimiter)
        query_dict = {}
        for part in parts:
            if ':' in part:
                key, value = part.split(':', 1)
                query_dict[key.strip()] = value.strip()
        return query_dict

    def retrieve(self, query, top_k=5, testbench=None):
        """
        Retrieve the top-k most relevant documents for a given query.

        :param query: The query string.
        :param method: The retrieval method, either 'Direct' or 'Associated'.
        :param top_k: The number of top documents to retrieve.
        :return: List of retrieved documents.
        """
        # print(f"@@@BREAKPOINT [retrieve ENTRY]: rag_type={self.rag_type}, top_k={top_k}")
        # print(f"@@@BREAKPOINT [retrieve ENTRY]: Total documents in pool: {len(self.retrieve_documents)}")
        # breakpoint()
        
        # Use Q-Tree inference if enabled (including enhanced retrieval mode)
        use_qtree = (
            (self.qtree_engine and self.rag_type == "Suggestions") or
            (getattr(FLAGS, 'retrieval_on_ranking', False) and self.rag_type == "Suggestions")
        )
        
        if use_qtree:
            print("\n=== QTREE RETRIEVAL ENTRY POINT ===")
            print(f"RAGAgent.retrieve() called with QTree inference enabled")
            print(f"Query length: {len(query)} chars")
            print(f"retrieval_on_ranking: {getattr(FLAGS, 'retrieval_on_ranking', False)}")
            # breakpoint()  # BREAKPOINT: Entry to QTree retrieval
            try:
                result = self.retrieve_with_qtree(query, top_k, testbench)
                # retrieve_with_qtree now returns dict with suggestions and initial_sva
                # Pass it through as-is so caller can handle appropriately
                return result
            except Exception as e:
                print(f"Q-Tree inference failed: {e}, falling back to regular retrieval")
        
        # Otherwise use regular retrieval
        if FLAGS.debug:
            print(f"@@@@@@@query: {query}")

        # Extract the relevant section from the query
        extracted_query = self.extract_section(query, FLAGS.similarity_str, full_string=True)
        # extracted_query = query

        start_retrieve_time = time.time()

        if isinstance(self.model, SentenceTransformer):
            query_embedding = self.model.encode([extracted_query], convert_to_tensor=True).cpu().numpy()
        elif isinstance(self.model, tuple):
            tokenizer, model = self.model
            tokens = tokenizer([extracted_query], padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                query_embedding = model(**tokens).last_hidden_state.mean(dim=1).cpu().numpy()
        else:
            query_embedding = self.model([extracted_query]).numpy()

        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        top_k_indices = similarities.argsort()[-(top_k + 1):][::-1]  # Get top (k+1) indices

        if not hasattr(FLAGS, 'filter_self'):
            retrieved_docs = [self.retrieve_documents[idx] for idx in top_k_indices[:top_k]]
        elif FLAGS.filter_self:
            # Skip the most similar document and return the next most similar ones
            retrieved_docs = [self.retrieve_documents[idx] for idx in top_k_indices[1:]]
        else:
            # Return the top-k most similar documents
            retrieved_docs = [self.retrieve_documents[idx] for idx in top_k_indices[:top_k]]

        if self.rag_type == 'Suggestions':
            if FLAGS.deduplication:
                retrieved_docs = self._deduplicate(retrieved_docs)

        end_retrieve_time = time.time()

        # Save stats
        saver.save_stats('retrieving_runtime', end_retrieve_time - start_retrieve_time)
        
        return retrieved_docs


    def _format_time(self, seconds):
        """
        Format the elapsed time into hours, minutes, and seconds.
        
        :param seconds: Elapsed time in seconds.
        :return: Formatted time string.
        """
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

    def _deduplicate(self, documents):
        """
        Deduplicate the list of documents.
        
        :param documents: List of documents to deduplicate.
        :return: List of deduplicated documents.
        """
        seen = set()
        deduplicated_docs = []
        for doc in documents:
            if doc not in seen:
                deduplicated_docs.append(doc)
                seen.add(doc)
        return deduplicated_docs

def main():
    # Example documents
    documents = [
        "This is the first document chunk.",
        "This is the second document chunk.",
        "Here is another piece of text.",
        "More data in this chunk.",
        "Final chunk of information."
    ] * 100
    
    # Choose an embedding model: 'sbert', 'use', or 'bert'
    embedding_model = 'sbert'
    
    # Initialize the RAGAgent
    rag_agent = RAGAgent(documents, embedding_model)
    
    # Example query
    query = "text"
    
    # Retrieve documents
    retrieved_docs = rag_agent.retrieve(query, top_k=2)
    
    # Print retrieved documents
    print("Retrieved Documents:")
    for doc in retrieved_docs:
        print(doc)

if __name__ == "__main__":
    main()
