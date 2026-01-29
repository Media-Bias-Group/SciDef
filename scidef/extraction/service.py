import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union

import dspy
from tqdm.asyncio import tqdm

from scidef.extraction.dataclass import ChunkMode
from scidef.extraction.extractor import (
    MultiStepExtractor,
    MultiStepFewShotExtractor,
    OnesStepExtractor,
    OnesStepFewShotExtractor,
)
from scidef.extraction.extractor.dspy_extraction import DSPyPaperExtractor
from scidef.grobid.service import extract_text_from_grobid
from scidef.model.dataclass import (
    ExtractionPrompt,
    ExtractionResult,
    PaperMetadata,
)
from scidef.utils import get_custom_colored_logger

logger = get_custom_colored_logger(__name__)


class ExtractionService:
    def __init__(
        self,
        extractor: Union[
            MultiStepExtractor,
            OnesStepExtractor,
            OnesStepFewShotExtractor,
            DSPyPaperExtractor,
            MultiStepFewShotExtractor,
        ],
        chunk_mode: ChunkMode = ChunkMode.PARAGRAPH,
        max_concurrent_papers=32,
    ) -> None:
        self.extractor = extractor
        self.chunk_mode = chunk_mode
        self.max_concurrent_papers = max_concurrent_papers
        self.is_dspy_extractor = isinstance(extractor, DSPyPaperExtractor)
        self.executor = None
        if self.is_dspy_extractor:
            self.executor = ThreadPoolExecutor(
                max_workers=max_concurrent_papers,
            )

            if max_concurrent_papers > 32:
                dspy.configure_cache(
                    enable_disk_cache=False,
                    enable_memory_cache=True,
                )
                logger.warning(
                    "Using in-memory caching for DSPy extractor with high concurrency "
                    "may lead to high memory usage and potential performance degradation due to disregarding previously cached results.",
                )

    async def extract_definition(
        self,
        paper_metadata: Optional[PaperMetadata] = None,
        chunks: Optional[List[str]] = None,
    ) -> Dict[str, List[Dict[str, str] | ExtractionResult]]:
        """Extract definitions from a single paper."""
        assert paper_metadata is not None or chunks is not None, (
            "Either paper_metadata or chunks must be provided",
        )
        paper_id = paper_metadata.paper_id if paper_metadata else "0"
        try:
            if chunks is None:
                assert paper_metadata is not None
                chunks = extract_text_from_grobid(
                    paper_metadata.xml_file_path,
                    self.chunk_mode,
                )
            definitions = []
            if not self.is_dspy_extractor:
                for chunk in chunks:
                    definition = await self.extractor.run(chunk)
                    if definition:
                        # assume: extractor returns List
                        definitions.extend(definition)
            else:
                assert isinstance(self.extractor, DSPyPaperExtractor)
                loop = asyncio.get_running_loop()

                logger.info(
                    f"[{paper_id}] Starting DSPy extraction on "
                    f"{len(chunks)} chunks (mode={self.chunk_mode})",
                )

                dspy_pred = await loop.run_in_executor(
                    self.executor,
                    lambda: self.extractor(
                        sections=chunks,
                    ),  # type: ignore DSPy extractor expects list of chunks
                )

                logger.info(
                    f"[{paper_id}] Finished DSPy extraction",
                )

                merged_json = json.loads(dspy_pred.merged_json)
                merged_json = [
                    {
                        "Term": item["term"],
                        "Definition": item["definition"],
                        "Type": item["type"],
                        "Context": item["context"],
                        "Input": chunks,
                    }
                    for item in merged_json
                ]
                definitions.extend(merged_json)

            if paper_metadata:
                return {paper_metadata.paper_id: definitions}
            else:
                return {"0": definitions}
        except Exception as e:
            logger.error(
                f"Definition extraction failed for {paper_id}: {e}",
            )
            return {
                paper_id: [
                    self._create_failed_result(
                        paper_id,
                        str(e),
                    ),
                ],
            }

    async def extract_definitions(
        self,
        paper_metadatas: List[PaperMetadata],
        sem: Optional[asyncio.Semaphore] = None,
    ) -> Dict[str, Union[List, ExtractionResult]]:
        sem = sem or asyncio.Semaphore(self.max_concurrent_papers)

        async def process_one(paper_metadata: PaperMetadata):
            async with sem:
                return await self.extract_definition(
                    paper_metadata=paper_metadata,
                )

        tasks = [
            asyncio.create_task(process_one(metadata))
            for metadata in paper_metadatas
        ]

        results = {}
        successful = 0
        failed = 0

        for future in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Extracting definitions",
            position=2,
            leave=False,
        ):
            result = await future
            results.update(result)

            # Track success/failure
            paper_id = list(result.keys())[0]
            if isinstance(result[paper_id], ExtractionResult):
                failed += 1
            else:
                successful += 1

        logger.info(
            f"Extraction complete: {successful} successful, {failed} failed "
            f"out of {len(paper_metadatas)} papers",
        )

        if self.executor is not None:
            self.executor.shutdown(wait=True)

        return results

    def _create_failed_result(
        self,
        paper_id: str,
        error_message: str,
    ) -> ExtractionResult:
        """Create a failed extraction result."""
        result = ExtractionResult(
            paper_id=paper_id,
            mode=ExtractionPrompt.EXTRACTIVE,
        )
        result.mark_failed(error_message)
        return result
