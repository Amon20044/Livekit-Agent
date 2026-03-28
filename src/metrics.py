"""
Non-blocking metrics collector for LiveKit agents with Prometheus.

This module provides an asyncio-based metrics collection system that:
- Updates Prometheus metrics asynchronously to avoid blocking the agent
- Supports dynamic configuration of models, providers, and pricing
- Uses a background worker queue for zero-latency metric updates
"""

import asyncio
import logging
import os
from typing import Optional, Dict, Any, Callable
from datetime import datetime

from prometheus_client import (
    Counter,
    Gauge,
    CollectorRegistry,
    multiprocess
)
from livekit.agents.metrics import (
    LLMMetrics,
    STTMetrics,
    TTSMetrics,
    VADMetrics,
    EOUMetrics,
    UsageCollector
)
from livekit.agents import vad
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout
from rich.live import Live
from rich import box

logger = logging.getLogger(__name__)
console = Console()


class MetricsCollector:
    """
    Async-first metrics collector for LiveKit agents.
    
    Features:
    - Non-blocking metric updates via asyncio queue
    - Dynamic pricing and model configuration
    - Multiprocess-safe Prometheus metrics
    - Automatic cost calculation
    
    Example:
        collector = MetricsCollector(
            agent_type="voice-agent",
            llm_config={
                "model": "gemini-2.0-flash-lite",
                "price_per_1k_input": 0.01,
                "price_per_1k_output": 0.03
            },
            stt_config={
                "provider": "assemblyai",
                "price_per_second": 0.0001
            },
            tts_config={
                "provider": "elevenlabs",
                "price_per_char": 0.000015
            }
            use_prometheus=true // false by default
        )
        
        # Start the async worker
        await collector.start()
        
        # Use in agent session
        session.on("metrics_collected", collector.handle_metrics)
    """
    
    def __init__(
        self,
        agent_type: str = "agent",
        llm_config: Optional[Dict[str, Any]] = None,
        stt_config: Optional[Dict[str, Any]] = None,
        tts_config: Optional[Dict[str, Any]] = None,
        registry: Optional[CollectorRegistry] = None,
        queue_size: int = 2000,
        prometheus_multiproc_dir: str = '/tmp/prometheus_multiproc',
        enable_table_logging: bool = True,
        enable_prometheus: bool = False,
    ):
        """
        Initialize the metrics collector.
        
        Args:
            agent_type: Name/type of the agent (used as label)
            llm_config: LLM configuration dict with keys:
                - model: str (e.g., "gemini-2.0-flash-lite")
                - price_per_1k_input: float (cost per 1K input tokens)
                - price_per_1k_output: float (cost per 1K output tokens)
            stt_config: STT configuration dict with keys:
                - provider: str (e.g., "assemblyai")
                - price_per_second: float (cost per second of audio)
            tts_config: TTS configuration dict with keys:
                - provider: str (e.g., "elevenlabs")
                - price_per_char: float (cost per character)
            registry: Prometheus registry (creates new if None)
            queue_size: Max size of async metrics queue
            prometheus_multiproc_dir: Directory for multiprocess metrics
            enable_table_logging: Enable colorful table output for latency analysis
        """
        self.agent_type = agent_type
        self.enable_table_logging = enable_table_logging
        self.enable_prometheus = enable_prometheus
        
        # Default configurations
        self.llm_config = llm_config or {
            "model": "gemini-2.5-flash-lite",
            "price_per_1k_input": 0.01,
            "price_per_1k_output": 0.03
        }
        
        self.stt_config = stt_config or {
            "provider": "assemblyai",
            "price_per_second": 0.0001
        }
        
        self.tts_config = tts_config or {
            "provider": "elevenlabs",
            "price_per_char": 0.000015
        }
        
        # Setup multiprocess directory only if Prometheus is enabled
        if enable_prometheus:
            os.environ['prometheus_multiproc_dir'] = prometheus_multiproc_dir
            
            # Create or use provided registry
            if registry is None:
                self.registry = CollectorRegistry()
                multiprocess.MultiProcessCollector(self.registry)
            else:
                self.registry = registry
            
            # Initialize Prometheus metrics
            self._init_prometheus_metrics()
        else:
            self.registry = None
        
        # Async queue for non-blocking updates
        self._metrics_queue: Optional[asyncio.Queue] = None
        self._metrics_worker_task: Optional[asyncio.Task] = None
        self._queue_size = queue_size
        
        # Usage collector for cost tracking
        self.usage_collector = UsageCollector()
        
        # Current turn tracking
        self.current_turn_metrics = {
            'turn_id': None,
            'eou_delay': None,  # External overhead (turn detection)
            'stt_duration': None,  # Core API 1: Speech-to-Text
            'llm_ttft': None,  # Core API 2: LLM inference
            'tts_ttfb': None,  # Core API 3: Text-to-Speech
        }
        self.turn_id_counter = 0
        
        logger.info(
            f"MetricsCollector initialized for agent_type={agent_type}, "
            f"llm={self.llm_config['model']}, "
            f"stt={self.stt_config['provider']}, "
            f"tts={self.tts_config['provider']}"
        )
    
    def _init_prometheus_metrics(self):
        """Initialize all Prometheus metrics with proper labels."""
        # Latency metrics
        self.llm_latency = Gauge(
            'livekit_llm_duration_ms',
            'LLM latency in milliseconds',
            ['model', 'agent_type'],
            registry=self.registry
        )
        
        self.stt_latency = Gauge(
            'livekit_stt_duration_ms',
            'Speech-to-text latency in milliseconds',
            ['provider', 'agent_type'],
            registry=self.registry
        )
        
        self.stt_transcription_delay = Gauge(
            'livekit_stt_transcription_delay_ms',
            'STT transcription delay in milliseconds',
            ['provider', 'agent_type'],
            registry=self.registry
        )
        
        self.tts_latency = Gauge(
            'livekit_tts_duration_ms',
            'Text-to-speech latency in milliseconds',
            ['provider', 'agent_type'],
            registry=self.registry
        )
        
        self.eou_latency = Gauge(
            'livekit_eou_delay_ms',
            'End-of-utterance delay in milliseconds',
            ['agent_type'],
            registry=self.registry
        )
        
        self.total_conversation_latency = Gauge(
            'livekit_total_conversation_latency_ms',
            'Current conversation latency in milliseconds',
            ['agent_type'],
            registry=self.registry
        )
        
        # Usage counters
        self.llm_tokens = Counter(
            'livekit_llm_tokens_total',
            'Total LLM tokens processed',
            ['type', 'model'],
            registry=self.registry
        )
        self.llm_tokens._multiprocess_mode = 'livesum'
        
        self.stt_duration = Counter(
            'livekit_stt_duration_seconds_total',
            'Total STT audio duration in seconds',
            ['provider'],
            registry=self.registry
        )
        self.stt_duration._multiprocess_mode = 'livesum'
        
        self.tts_chars = Counter(
            'livekit_tts_chars_total',
            'Total TTS characters processed',
            ['provider'],
            registry=self.registry
        )
        self.tts_chars._multiprocess_mode = 'livesum'
        
        self.total_tokens = Counter(
            'livekit_total_tokens_total',
            'Total tokens processed',
            registry=self.registry
        )
        self.total_tokens._multiprocess_mode = 'livesum'
        
        self.conversation_turns = Counter(
            'livekit_conversation_turns_total',
            'Number of conversation turns',
            ['agent_type', 'room'],
            registry=self.registry
        )
        self.conversation_turns._multiprocess_mode = 'livesum'
        
        self.active_conversations = Gauge(
            'livekit_active_conversations',
            'Number of active conversations',
            ['agent_type'],
            multiprocess_mode='liveall',
            registry=self.registry
        )
        
        # Cost metrics
        self.llm_cost = Gauge(
            'livekit_llm_cost_total',
            'Total LLM cost in USD',
            ['model'],
            registry=self.registry
        )
        self.llm_cost._multiprocess_mode = 'liveall'
        
        self.stt_cost = Gauge(
            'livekit_stt_cost_total',
            'Total STT cost in USD',
            ['provider'],
            registry=self.registry
        )
        self.stt_cost._multiprocess_mode = 'liveall'
        
        self.tts_cost = Gauge(
            'livekit_tts_cost_total',
            'Total TTS cost in USD',
            ['provider'],
            registry=self.registry
        )
        self.tts_cost._multiprocess_mode = 'liveall'
        
        # Initialize with zero values
        self._initialize_default_values()
    
    def _initialize_default_values(self):
        """Set initial zero values for all metrics."""
        llm_model = self.llm_config['model']
        stt_provider = self.stt_config['provider']
        tts_provider = self.tts_config['provider']
        
        # Latencies
        self.llm_latency.labels(model=llm_model, agent_type=self.agent_type).set(0)
        self.stt_latency.labels(provider=stt_provider, agent_type=self.agent_type).set(0)
        self.stt_transcription_delay.labels(provider=stt_provider, agent_type=self.agent_type).set(0)
        self.tts_latency.labels(provider=tts_provider, agent_type=self.agent_type).set(0)
        self.eou_latency.labels(agent_type=self.agent_type).set(0)
        self.total_conversation_latency.labels(agent_type=self.agent_type).set(0)
        
        # Tokens and usage
        self.llm_tokens.labels(type='prompt', model=llm_model).inc(0)
        self.llm_tokens.labels(type='completion', model=llm_model).inc(0)
        self.stt_duration.labels(provider=stt_provider).inc(0)
        self.tts_chars.labels(provider=tts_provider).inc(0)
        self.total_tokens.inc(0)
        
        # Costs
        self.llm_cost.labels(model=llm_model).set(0)
        self.stt_cost.labels(provider=stt_provider).set(0)
        self.tts_cost.labels(provider=tts_provider).set(0)
        
        logger.info("Initialized all metrics with default zero values")
    
    def _get_latency_color(self, latency_ms: float, thresholds: Dict[str, float]) -> str:
        """
        Get color based on latency thresholds.
        
        Args:
            latency_ms: Latency value in milliseconds
            thresholds: Dict with 'good', 'warning', 'critical' keys
        
        Returns:
            Color name for rich formatting
        """
        if latency_ms <= thresholds.get('good', 100):
            return "green"
        elif latency_ms <= thresholds.get('warning', 300):
            return "yellow"
        elif latency_ms <= thresholds.get('critical', 500):
            return "red"
        else:
            return "bright_red"
    
    def _create_latency_table(self) -> Table:
        """Create a rich table with current latency metrics."""
        table = Table(
            title="🚀 LiveKit Agent Latency Analysis",
            show_header=True,
            header_style="bold magenta",
            border_style="cyan",
            show_lines=True
        )
        
        table.add_column("Component", style="cyan", width=20)
        table.add_column("Latency (ms)", justify="right", width=15)
        table.add_column("Status", justify="center", width=12)
        table.add_column("Details", style="dim", width=30)
        
        # Define thresholds for different components
        thresholds = {
            'stt': {'good': 100, 'warning': 300, 'critical': 600},
            'llm': {'good': 200, 'warning': 500, 'critical': 1000},
            'tts': {'good': 150, 'warning': 400, 'critical': 800},
            'total_core': {'good': 450, 'warning': 900, 'critical': 1500},
            'eou': {'good': 100, 'warning': 200, 'critical': 300}
        }
        
        # Get current values
        stt_ms = self.current_turn_metrics.get('stt_duration', 0) * 1000 if self.current_turn_metrics.get('stt_duration') else 0
        llm_ms = self.current_turn_metrics.get('llm_ttft', 0) * 1000 if self.current_turn_metrics.get('llm_ttft') else 0
        tts_ms = self.current_turn_metrics.get('tts_ttfb', 0) * 1000 if self.current_turn_metrics.get('tts_ttfb') else 0
        eou_ms = self.current_turn_metrics.get('eou_delay', 0) * 1000 if self.current_turn_metrics.get('eou_delay') else 0
        
        # Calculate totals
        total_core_ms = stt_ms + llm_ms + tts_ms  # Core API calls only
        total_with_overhead_ms = total_core_ms + eou_ms  # Including EOU overhead
        
        # STT (Core API 1)
        stt_color = self._get_latency_color(stt_ms, thresholds['stt'])
        stt_status = "✓ GOOD" if stt_ms <= thresholds['stt']['good'] else ("⚠ WARN" if stt_ms <= thresholds['stt']['warning'] else "✗ SLOW")
        table.add_row(
            "🎤 STT API",
            f"[{stt_color}]{stt_ms:.1f}[/{stt_color}]",
            f"[{stt_color}]{stt_status}[/{stt_color}]",
            f"Provider: {self.stt_config['provider']}"
        )
        
        # LLM (Core API 2)
        llm_color = self._get_latency_color(llm_ms, thresholds['llm'])
        llm_status = "✓ GOOD" if llm_ms <= thresholds['llm']['good'] else ("⚠ WARN" if llm_ms <= thresholds['llm']['warning'] else "✗ SLOW")
        table.add_row(
            f"🧠 LLM API (TTFT)",
            f"[{llm_color}]{llm_ms:.1f}[/{llm_color}]",
            f"[{llm_color}]{llm_status}[/{llm_color}]",
            f"Model: {self.llm_config['model']}"
        )
        
        # TTS (Core API 3)
        tts_color = self._get_latency_color(tts_ms, thresholds['tts'])
        tts_status = "✓ GOOD" if tts_ms <= thresholds['tts']['good'] else ("⚠ WARN" if tts_ms <= thresholds['tts']['warning'] else "✗ SLOW")
        table.add_row(
            f"🔊 TTS API (TTFB)",
            f"[{tts_color}]{tts_ms:.1f}[/{tts_color}]",
            f"[{tts_color}]{tts_status}[/{tts_color}]",
            f"Provider: {self.tts_config['provider']}"
        )
        
        # Separator
        table.add_row("", "", "", "", end_section=True)
        
        # Total Core APIs
        core_color = self._get_latency_color(total_core_ms, thresholds['total_core'])
        core_status = "✓ GOOD" if total_core_ms <= thresholds['total_core']['good'] else ("⚠ WARN" if total_core_ms <= thresholds['total_core']['warning'] else "✗ SLOW")
        table.add_row(
            "⚡ CORE APIs TOTAL",
            f"[bold {core_color}]{total_core_ms:.1f}[/bold {core_color}]",
            f"[bold {core_color}]{core_status}[/bold {core_color}]",
            f"STT + LLM + TTS",
            style="bold"
        )
        
        # EOU Overhead (external)
        eou_color = self._get_latency_color(eou_ms, thresholds['eou'])
        eou_status = "✓ GOOD" if eou_ms <= thresholds['eou']['good'] else ("⚠ WARN" if eou_ms <= thresholds['eou']['warning'] else "✗ SLOW")
        table.add_row(
            "🎯 EOU Overhead",
            f"[{eou_color}]{eou_ms:.1f}[/{eou_color}]",
            f"[{eou_color}]{eou_status}[/{eou_color}]",
            f"Turn detection (external)",
            style="dim"
        )
        
        # Grand Total (with overhead)
        table.add_row(
            "🏁 TOTAL (with EOU)",
            f"[bold white]{total_with_overhead_ms:.1f}[/bold white]",
            "",
            f"Turn #{self.current_turn_metrics.get('turn_id', 0)}",
            style="dim"
        )
        
        return table
    
    def _create_usage_table(self) -> Table:
        """Create a rich table with usage and cost metrics."""
        table = Table(
            title="💰 Usage & Cost Tracking",
            show_header=True,
            header_style="bold yellow",
            border_style="green",
            show_lines=True
        )
        
        table.add_column("Service", style="yellow", width=20)
        table.add_column("Usage", justify="right", width=20)
        table.add_column("Cost (USD)", justify="right", width=15)
        table.add_column("Provider/Model", style="dim", width=25)
        
        try:
            summary = self.usage_collector.get_summary()
            
            # LLM
            llm_prompt = getattr(summary, 'llm_prompt_tokens', 0)
            llm_completion = getattr(summary, 'llm_completion_tokens', 0)
            llm_total = llm_prompt + llm_completion
            llm_cost = (
                llm_prompt * self.llm_config['price_per_1k_input'] / 1000 +
                llm_completion * self.llm_config['price_per_1k_output'] / 1000
            )
            
            table.add_row(
                "🧠 LLM Tokens",
                f"{llm_total:,} ({llm_prompt:,} in / {llm_completion:,} out)",
                f"[green]${llm_cost:.6f}[/green]",
                self.llm_config['model']
            )
            
            # STT
            stt_duration = getattr(summary, 'stt_audio_duration', 0)
            stt_cost = stt_duration * self.stt_config['price_per_second']
            
            table.add_row(
                "🎤 STT Audio",
                f"{stt_duration:.2f} seconds",
                f"[green]${stt_cost:.6f}[/green]",
                self.stt_config['provider']
            )
            
            # TTS
            tts_chars = getattr(summary, 'tts_characters_count', 0)
            tts_cost = tts_chars * self.tts_config['price_per_char']
            
            table.add_row(
                "🔊 TTS Characters",
                f"{tts_chars:,} chars",
                f"[green]${tts_cost:.6f}[/green]",
                self.tts_config['provider']
            )
            
            # Total
            total_cost = llm_cost + stt_cost + tts_cost
            table.add_row(
                "💵 TOTAL COST",
                "",
                f"[bold yellow]${total_cost:.6f}[/bold yellow]",
                "Session total",
                style="bold"
            )
            
        except Exception as e:
            logger.error(f"Error creating usage table: {e}")
        
        return table
    
    def log_metrics_table(self):
        """Display colorful tables with current metrics."""
        if not self.enable_table_logging:
            return
        
        try:
            console.print("\n")
            console.print(self._create_latency_table())
            console.print("\n")
            console.print(self._create_usage_table())
            console.print("\n")
            
        except Exception as e:
            logger.error(f"Error displaying metrics table: {e}")
    
    def log_latency_breakdown(self):
        """Log a detailed breakdown of latency components."""
        if not self.enable_table_logging:
            return
        
        try:
            stt_ms = self.current_turn_metrics.get('stt_duration', 0) * 1000 if self.current_turn_metrics.get('stt_duration') else 0
            llm_ms = self.current_turn_metrics.get('llm_ttft', 0) * 1000 if self.current_turn_metrics.get('llm_ttft') else 0
            tts_ms = self.current_turn_metrics.get('tts_ttfb', 0) * 1000 if self.current_turn_metrics.get('tts_ttfb') else 0
            eou_ms = self.current_turn_metrics.get('eou_delay', 0) * 1000 if self.current_turn_metrics.get('eou_delay') else 0
            
            total_core_ms = stt_ms + llm_ms + tts_ms  # Core API calls
            total_with_overhead_ms = total_core_ms + eou_ms  # Including overhead
            
            if total_core_ms == 0:
                return
            
            # Create a breakdown panel
            breakdown_text = Text()
            breakdown_text.append("📊 Core API Latency Breakdown:\n", style="bold cyan")
            breakdown_text.append("   (STT → LLM → TTS)\n\n", style="dim")
            
            # Calculate percentages based on CORE APIs only
            stt_pct = (stt_ms / total_core_ms * 100) if total_core_ms > 0 else 0
            llm_pct = (llm_ms / total_core_ms * 100) if total_core_ms > 0 else 0
            tts_pct = (tts_ms / total_core_ms * 100) if total_core_ms > 0 else 0
            
            breakdown_text.append(f"  1️⃣  STT API:        {stt_ms:6.1f}ms  ({stt_pct:5.1f}%)  ", style="yellow")
            breakdown_text.append("█" * int(stt_pct / 2), style="yellow")
            breakdown_text.append("\n")
            breakdown_text.append(f"      └─ Speech-to-Text (Provider: {self.stt_config['provider']})\n", style="dim yellow")
            
            breakdown_text.append(f"  2️⃣  LLM API:        {llm_ms:6.1f}ms  ({llm_pct:5.1f}%)  ", style="magenta")
            breakdown_text.append("█" * int(llm_pct / 2), style="magenta")
            breakdown_text.append("\n")
            breakdown_text.append(f"      └─ Language Model TTFT ({self.llm_config['model']})\n", style="dim magenta")
            
            breakdown_text.append(f"  3️⃣  TTS API:        {tts_ms:6.1f}ms  ({tts_pct:5.1f}%)  ", style="cyan")
            breakdown_text.append("█" * int(tts_pct / 2), style="cyan")
            breakdown_text.append("\n")
            breakdown_text.append(f"      └─ Text-to-Speech TTFB ({self.tts_config['provider']})\n\n", style="dim cyan")
            
            breakdown_text.append(f"  ⚡ CORE APIs:      {total_core_ms:6.1f}ms  (100.0%)  ", style="bold green")
            breakdown_text.append("█" * 50, style="bold green")
            breakdown_text.append("\n\n")
            
            # Show overhead separately
            if eou_ms > 0:
                overhead_pct = (eou_ms / total_with_overhead_ms * 100) if total_with_overhead_ms > 0 else 0
                breakdown_text.append(f"  🎯 EOU Overhead:   {eou_ms:6.1f}ms  ({overhead_pct:5.1f}% of total)  ", style="dim white")
                breakdown_text.append("░" * int(overhead_pct / 2), style="dim white")
                breakdown_text.append("\n")
                breakdown_text.append("      └─ Turn detection (external)\n\n", style="dim")
                
                breakdown_text.append(f"  🏁 TOTAL:          {total_with_overhead_ms:6.1f}ms  ", style="bold white")
                breakdown_text.append("(Core + Overhead)", style="dim")
            
            panel = Panel(
                breakdown_text,
                title=f"[bold]Turn #{self.current_turn_metrics.get('turn_id', 0)} - API Performance[/bold]",
                border_style="bright_blue",
                padding=(1, 2)
            )
            
            console.print(panel)
            console.print("\n")
            
        except Exception as e:
            logger.error(f"Error displaying latency breakdown: {e}")
    
    async def start(self):
        """Start the async metrics worker."""
        if self._metrics_queue is None:
            self._metrics_queue = asyncio.Queue(maxsize=self._queue_size)
        
        if self._metrics_worker_task is None:
            loop = asyncio.get_running_loop()
            self._metrics_worker_task = loop.create_task(self._metrics_worker())
            logger.info("Started async metrics worker")
    
    async def _metrics_worker(self):
        """Background worker that processes metric updates asynchronously."""
        assert self._metrics_queue is not None
        
        while True:
            op = await self._metrics_queue.get()
            try:
                if asyncio.iscoroutine(op):
                    await op
                elif callable(op):
                    # Run sync operations in thread pool to avoid blocking
                    await asyncio.to_thread(op)
                else:
                    logger.warning(f"Unknown metric operation type: {type(op)}")
            except Exception:
                logger.exception("Error processing metric operation")
            finally:
                try:
                    self._metrics_queue.task_done()
                except Exception:
                    pass
    
    def enqueue_metric_op(self, op: Callable):
        """
        Enqueue a metric operation for async processing.
        
        This is non-blocking - if queue is full, the operation is dropped
        with a warning to avoid slowing down the agent.
        """
        # Skip if Prometheus is disabled
        if not self.enable_prometheus:
            return
            
        if self._metrics_queue is None:
            # Fallback: execute synchronously if worker not started
            try:
                if callable(op):
                    op()
                elif asyncio.iscoroutine(op):
                    asyncio.run(op)
            except Exception:
                logger.exception("Failed to execute metric op synchronously")
            return
        
        try:
            self._metrics_queue.put_nowait(op)
        except asyncio.QueueFull:
            logger.warning("Metrics queue full - dropping metric update")
    
    def start_new_turn(self, room: str = 'unknown'):
        """Start tracking a new conversation turn."""
        self.turn_id_counter += 1
        self.current_turn_metrics['turn_id'] = self.turn_id_counter
        self.current_turn_metrics['eou_delay'] = None
        self.current_turn_metrics['stt_duration'] = None
        self.current_turn_metrics['llm_ttft'] = None
        self.current_turn_metrics['tts_ttfb'] = None
        
        # Increment turn counter
        self.enqueue_metric_op(
            lambda: self.conversation_turns.labels(
                agent_type=self.agent_type,
                room=room
            ).inc()
        )
        
        logger.debug(f"Started new turn {self.turn_id_counter} in room {room}")
        return self.turn_id_counter
    
    def calculate_total_latency(self):
        """Calculate and update total conversation latency."""
        # Check if all CORE APIs have reported
        if all(self.current_turn_metrics[k] is not None 
               for k in ['stt_duration', 'llm_ttft', 'tts_ttfb']):
            
            # Convert to milliseconds
            stt_ms = self.current_turn_metrics['stt_duration'] * 1000
            llm_ms = self.current_turn_metrics['llm_ttft'] * 1000
            tts_ms = self.current_turn_metrics['tts_ttfb'] * 1000
            eou_ms = (self.current_turn_metrics.get('eou_delay') or 0) * 1000
            
            # Core API total (what we actually care about)
            total_core_ms = int(stt_ms + llm_ms + tts_ms)
            
            # Total with overhead
            total_with_overhead_ms = int(total_core_ms + eou_ms)
            
            # Update metric asynchronously (store core APIs total)
            self.enqueue_metric_op(
                lambda: self.total_conversation_latency.labels(
                    agent_type=self.agent_type
                ).set(total_core_ms)
            )
            
            logger.info(
                f"Core APIs: {total_core_ms}ms (STT={int(stt_ms)}, LLM={int(llm_ms)}, TTS={int(tts_ms)}) "
                f"+ EOU Overhead={int(eou_ms)}ms = Total {total_with_overhead_ms}ms"
            )
            
            # Display colorful tables
            self.log_latency_breakdown()
            self.log_metrics_table()
    
    def handle_metrics(self, ev):
        """
        Handle metrics collected events from LiveKit AgentSession.
        
        This is the main entry point - register this as a callback:
            session.on("metrics_collected", collector.handle_metrics)
        """
        from livekit.agents import MetricsCollectedEvent
        
        if not isinstance(ev, MetricsCollectedEvent):
            return
        
        # Collect usage for cost calculation
        try:
            self.usage_collector.collect(ev.metrics)
            self._update_usage_metrics()
        except Exception as e:
            logger.error(f"Error collecting usage metrics: {e}")
        
        # Process specific metric types
        if isinstance(ev.metrics, LLMMetrics):
            self._handle_llm_metrics(ev.metrics)
        elif isinstance(ev.metrics, STTMetrics):
            self._handle_stt_metrics(ev.metrics)
        elif isinstance(ev.metrics, TTSMetrics):
            self._handle_tts_metrics(ev.metrics)
        elif isinstance(ev.metrics, EOUMetrics):
            self._handle_eou_metrics(ev.metrics)
        elif isinstance(ev.metrics, VADMetrics):
            # VADMetrics is actually a VADEvent
            self._handle_vad_metrics(ev.metrics)
        else:
            logger.debug(f"Unknown metrics type: {type(ev.metrics)}")
    
    def _update_usage_metrics(self):
        """Update usage counters and costs from collected metrics."""
        # Skip if Prometheus is disabled
        if not self.enable_prometheus:
            return
            
        try:
            summary = self.usage_collector.get_summary()
            llm_model = self.llm_config['model']
            stt_provider = self.stt_config['provider']
            tts_provider = self.tts_config['provider']
            
            # Get current values
            current_prompt = self.llm_tokens.labels(type='prompt', model=llm_model)._value.get() or 0
            current_completion = self.llm_tokens.labels(type='completion', model=llm_model)._value.get() or 0
            current_stt = self.stt_duration.labels(provider=stt_provider)._value.get() or 0
            current_tts = self.tts_chars.labels(provider=tts_provider)._value.get() or 0
            
            # Update tokens
            if hasattr(summary, 'llm_prompt_tokens'):
                new_val = summary.llm_prompt_tokens
                if new_val > current_prompt:
                    delta = new_val - current_prompt
                    self.enqueue_metric_op(
                        lambda d=delta: self.llm_tokens.labels(type='prompt', model=llm_model).inc(d)
                    )
            
            if hasattr(summary, 'llm_completion_tokens'):
                new_val = summary.llm_completion_tokens
                if new_val > current_completion:
                    delta = new_val - current_completion
                    self.enqueue_metric_op(
                        lambda d=delta: self.llm_tokens.labels(type='completion', model=llm_model).inc(d)
                    )
            
            # Update STT duration
            if hasattr(summary, 'stt_audio_duration'):
                new_val = summary.stt_audio_duration
                if new_val > current_stt:
                    delta = new_val - current_stt
                    self.enqueue_metric_op(
                        lambda d=delta: self.stt_duration.labels(provider=stt_provider).inc(d)
                    )
            
            # Update TTS characters
            if hasattr(summary, 'tts_characters_count'):
                new_val = summary.tts_characters_count
                if new_val > current_tts:
                    delta = new_val - current_tts
                    self.enqueue_metric_op(
                        lambda d=delta: self.tts_chars.labels(provider=tts_provider).inc(d)
                    )
            
            # Calculate and update costs
            llm_cost = (
                getattr(summary, 'llm_prompt_tokens', 0) * self.llm_config['price_per_1k_input'] / 1000 +
                getattr(summary, 'llm_completion_tokens', 0) * self.llm_config['price_per_1k_output'] / 1000
            )
            stt_cost = getattr(summary, 'stt_audio_duration', 0) * self.stt_config['price_per_second']
            tts_cost = getattr(summary, 'tts_characters_count', 0) * self.tts_config['price_per_char']
            
            self.enqueue_metric_op(lambda c=llm_cost: self.llm_cost.labels(model=llm_model).set(c))
            self.enqueue_metric_op(lambda c=stt_cost: self.stt_cost.labels(provider=stt_provider).set(c))
            self.enqueue_metric_op(lambda c=tts_cost: self.tts_cost.labels(provider=tts_provider).set(c))
            
        except Exception as e:
            logger.error(f"Error updating usage metrics: {e}")
    
    def _handle_llm_metrics(self, metrics: LLMMetrics):
        """Handle LLM-specific metrics."""
        llm_model = self.llm_config['model']
        
        # Log detailed LLM metrics table
        if self.enable_table_logging:
            self._log_llm_metrics_table(metrics)
        
        if hasattr(metrics, 'ttft') and metrics.ttft is not None:
            self.current_turn_metrics['llm_ttft'] = metrics.ttft
            ttft_ms = metrics.ttft * 1000
            self.enqueue_metric_op(
                lambda t=ttft_ms: self.llm_latency.labels(
                    model=llm_model,
                    agent_type=self.agent_type
                ).set(t)
            )
            
            self.calculate_total_latency()
        
        if hasattr(metrics, 'total_tokens') and metrics.total_tokens is not None:
            self.enqueue_metric_op(
                lambda t=metrics.total_tokens: self.total_tokens.inc(t)
            )
    
    def _log_llm_metrics_table(self, metrics: LLMMetrics):
        """Display detailed LLM metrics table."""
        table = Table(
            title="[bold blue]🧠 LLM Metrics Report[/bold blue]",
            box=box.ROUNDED,
            highlight=True,
            show_header=True,
            header_style="bold cyan"
        )
        
        table.add_column("Metric", style="bold green", width=25)
        table.add_column("Value", style="yellow", width=40)
        
        timestamp = datetime.fromtimestamp(metrics.timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] if hasattr(metrics, 'timestamp') else 'N/A'
        
        table.add_row("Type", str(getattr(metrics, 'type', 'N/A')))
        table.add_row("Label", str(getattr(metrics, 'label', 'N/A')))
        table.add_row("Request ID", str(getattr(metrics, 'request_id', 'N/A')))
        table.add_row("Timestamp", timestamp)
        
        # Duration
        duration = getattr(metrics, 'duration', None)
        if duration is not None:
            color = self._get_latency_color(duration * 1000, {'good': 500, 'warning': 1000, 'critical': 2000})
            table.add_row("Duration", f"[{color}]{duration:.4f}s ({duration*1000:.1f}ms)[/{color}]")
        else:
            table.add_row("Duration", "N/A")
        
        # TTFT
        ttft = getattr(metrics, 'ttft', None)
        if ttft is not None:
            color = self._get_latency_color(ttft * 1000, {'good': 200, 'warning': 500, 'critical': 1000})
            table.add_row("Time to First Token", f"[{color}]{ttft:.4f}s ({ttft*1000:.1f}ms)[/{color}]")
        else:
            table.add_row("Time to First Token", "N/A")
        
        table.add_row("Cancelled", "✓" if getattr(metrics, 'cancelled', False) else "✗")
        table.add_row("Completion Tokens", str(getattr(metrics, 'completion_tokens', 'N/A')))
        table.add_row("Prompt Tokens", str(getattr(metrics, 'prompt_tokens', 'N/A')))
        table.add_row("Total Tokens", str(getattr(metrics, 'total_tokens', 'N/A')))
        
        tokens_per_second = getattr(metrics, 'tokens_per_second', None)
        if tokens_per_second is not None:
            table.add_row("Tokens/Second", f"{tokens_per_second:.2f}")
        else:
            table.add_row("Tokens/Second", "N/A")
        
        console.print("\n")
        console.print(table)
        console.print("\n")
    
    def _handle_stt_metrics(self, metrics: STTMetrics):
        """Handle STT-specific metrics."""
        stt_provider = self.stt_config['provider']
        
        # Log detailed STT metrics table
        if self.enable_table_logging:
            self._log_stt_metrics_table(metrics)
        
        if hasattr(metrics, 'duration') and metrics.duration is not None:
            # Store STT duration for core API tracking
            self.current_turn_metrics['stt_duration'] = metrics.duration
            
            duration_ms = metrics.duration * 1000
            self.enqueue_metric_op(
                lambda d=duration_ms: self.stt_latency.labels(
                    provider=stt_provider,
                    agent_type=self.agent_type
                ).set(d)
            )
            
            # Try to calculate total if all metrics available
            self.calculate_total_latency()
    
    def _log_stt_metrics_table(self, metrics: STTMetrics):
        """Display detailed STT metrics table."""
        table = Table(
            title="[bold blue]🎤 STT (Speech-to-Text) API Report[/bold blue]",
            box=box.ROUNDED,
            highlight=True,
            show_header=True,
            header_style="bold cyan"
        )
        
        table.add_column("Metric", style="bold green", width=25)
        table.add_column("Value", style="yellow", width=40)
        
        timestamp = datetime.fromtimestamp(metrics.timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] if hasattr(metrics, 'timestamp') else 'N/A'
        
        table.add_row("Type", str(getattr(metrics, 'type', 'N/A')))
        table.add_row("Label", str(getattr(metrics, 'label', 'N/A')))
        table.add_row("Request ID", str(getattr(metrics, 'request_id', 'N/A')))
        table.add_row("Timestamp", timestamp)
        
        # STT API Duration (Core metric)
        duration = getattr(metrics, 'duration', None)
        if duration is not None:
            color = self._get_latency_color(duration * 1000, {'good': 100, 'warning': 300, 'critical': 600})
            table.add_row("STT API Duration", f"[bold {color}]{duration:.4f}s ({duration*1000:.1f}ms)[/bold {color}]")
        else:
            table.add_row("STT API Duration", "N/A")
        
        # Audio Duration
        audio_duration = getattr(metrics, 'audio_duration', None)
        if audio_duration is not None:
            table.add_row("Audio Duration", f"{audio_duration:.4f}s ({audio_duration*1000:.1f}ms)")
        else:
            table.add_row("Audio Duration", "N/A")
        
        table.add_row("Speech ID", str(getattr(metrics, 'speech_id', 'N/A')))
        table.add_row("Error", str(getattr(metrics, 'error', 'None')))
        table.add_row("Streamed", "✓" if getattr(metrics, 'streamed', False) else "✗")
        
        console.print("\n")
        console.print(table)
        console.print("\n")
    
    def _handle_tts_metrics(self, metrics: TTSMetrics):
        """Handle TTS-specific metrics."""
        tts_provider = self.tts_config['provider']
        
        # Log detailed TTS metrics table
        if self.enable_table_logging:
            self._log_tts_metrics_table(metrics)
        
        if hasattr(metrics, 'ttfb') and metrics.ttfb is not None:
            self.current_turn_metrics['tts_ttfb'] = metrics.ttfb
            ttfb_ms = metrics.ttfb * 1000
            self.enqueue_metric_op(
                lambda t=ttfb_ms: self.tts_latency.labels(
                    provider=tts_provider,
                    agent_type=self.agent_type
                ).set(t)
            )
            
            self.calculate_total_latency()
    
    def _log_tts_metrics_table(self, metrics: TTSMetrics):
        """Display detailed TTS metrics table."""
        table = Table(
            title="[bold blue]🔊 TTS Metrics Report[/bold blue]",
            box=box.ROUNDED,
            highlight=True,
            show_header=True,
            header_style="bold cyan"
        )
        
        table.add_column("Metric", style="bold green", width=25)
        table.add_column("Value", style="yellow", width=40)
        
        timestamp = datetime.fromtimestamp(metrics.timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] if hasattr(metrics, 'timestamp') else 'N/A'
        
        table.add_row("Type", str(getattr(metrics, 'type', 'N/A')))
        table.add_row("Label", str(getattr(metrics, 'label', 'N/A')))
        table.add_row("Request ID", str(getattr(metrics, 'request_id', 'N/A')))
        table.add_row("Timestamp", timestamp)
        
        # TTFB
        ttfb = getattr(metrics, 'ttfb', None)
        if ttfb is not None:
            color = self._get_latency_color(ttfb * 1000, {'good': 150, 'warning': 400, 'critical': 800})
            table.add_row("TTFB", f"[{color}]{ttfb:.4f}s ({ttfb*1000:.1f}ms)[/{color}]")
        else:
            table.add_row("TTFB", "N/A")
        
        # Duration
        duration = getattr(metrics, 'duration', None)
        if duration is not None:
            table.add_row("Duration", f"{duration:.4f}s ({duration*1000:.1f}ms)")
        else:
            table.add_row("Duration", "N/A")
        
        # Audio Duration
        audio_duration = getattr(metrics, 'audio_duration', None)
        if audio_duration is not None:
            table.add_row("Audio Duration", f"{audio_duration:.4f}s ({audio_duration*1000:.1f}ms)")
        else:
            table.add_row("Audio Duration", "N/A")
        
        table.add_row("Cancelled", "✓" if getattr(metrics, 'cancelled', False) else "✗")
        table.add_row("Characters Count", str(getattr(metrics, 'characters_count', 'N/A')))
        table.add_row("Streamed", "✓" if getattr(metrics, 'streamed', False) else "✗")
        table.add_row("Speech ID", str(getattr(metrics, 'speech_id', 'N/A')))
        table.add_row("Error", str(getattr(metrics, 'error', 'None')))
        
        console.print("\n")
        console.print(table)
        console.print("\n")
    
    def _handle_eou_metrics(self, metrics: EOUMetrics):
        """Handle end-of-utterance metrics."""
        stt_provider = self.stt_config['provider']
        
        # Log detailed EOU metrics table
        if self.enable_table_logging:
            self._log_eou_metrics_table(metrics)
        
        if hasattr(metrics, 'end_of_utterance_delay') and metrics.end_of_utterance_delay is not None:
            self.current_turn_metrics['eou_delay'] = metrics.end_of_utterance_delay
            delay_ms = metrics.end_of_utterance_delay * 1000
            self.enqueue_metric_op(
                lambda d=delay_ms: self.eou_latency.labels(
                    agent_type=self.agent_type
                ).set(d)
            )
            
            self.calculate_total_latency()
        
        if hasattr(metrics, 'transcription_delay') and metrics.transcription_delay is not None:
            trans_delay_ms = metrics.transcription_delay * 1000
            self.enqueue_metric_op(
                lambda d=trans_delay_ms: self.stt_transcription_delay.labels(
                    provider=stt_provider,
                    agent_type=self.agent_type
                ).set(d)
            )
    
    def _log_eou_metrics_table(self, metrics: EOUMetrics):
        """Display detailed End of Utterance metrics table."""
        table = Table(
            title="[bold blue]🎯 End of Utterance Metrics Report[/bold blue]",
            box=box.ROUNDED,
            highlight=True,
            show_header=True,
            header_style="bold cyan"
        )
        
        table.add_column("Metric", style="bold green", width=25)
        table.add_column("Value", style="yellow", width=40)
        
        timestamp = datetime.fromtimestamp(metrics.timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] if hasattr(metrics, 'timestamp') else 'N/A'
        
        table.add_row("Type", str(getattr(metrics, 'type', 'N/A')))
        table.add_row("Label", str(getattr(metrics, 'label', 'N/A')))
        table.add_row("Timestamp", timestamp)
        
        # EOU Delay
        eou_delay = getattr(metrics, 'end_of_utterance_delay', None)
        if eou_delay is not None:
            color = self._get_latency_color(eou_delay * 1000, {'good': 100, 'warning': 200, 'critical': 300})
            table.add_row("End of Utterance Delay", f"[{color}]{eou_delay:.4f}s ({eou_delay*1000:.1f}ms)[/{color}]")
        else:
            table.add_row("End of Utterance Delay", "N/A")
        
        # Transcription Delay
        trans_delay = getattr(metrics, 'transcription_delay', None)
        if trans_delay is not None:
            color = self._get_latency_color(trans_delay * 1000, {'good': 100, 'warning': 300, 'critical': 600})
            table.add_row("Transcription Delay", f"[{color}]{trans_delay:.4f}s ({trans_delay*1000:.1f}ms)[/{color}]")
        else:
            table.add_row("Transcription Delay", "N/A")
        
        table.add_row("Speech ID", str(getattr(metrics, 'speech_id', 'N/A')))
        table.add_row("Error", str(getattr(metrics, 'error', 'None')))
        
        console.print("\n")
        console.print(table)
        console.print("\n")
    
    def _handle_vad_metrics(self, event: vad.VADEvent):
        """Handle VAD event metrics."""
        # Log detailed VAD metrics table
        if self.enable_table_logging:
            self._log_vad_metrics_table(event)
    
    def _log_vad_metrics_table(self, event: vad.VADEvent):
        """Display detailed VAD metrics table."""
        table = Table(
            title="[bold blue]🎙️ VAD Event Metrics Report[/bold blue]",
            box=box.ROUNDED,
            highlight=True,
            show_header=True,
            header_style="bold cyan"
        )
        
        table.add_column("Metric", style="bold green", width=25)
        table.add_column("Value", style="yellow", width=40)
        
        timestamp = datetime.fromtimestamp(event.timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] if hasattr(event, 'timestamp') else 'N/A'
        
        table.add_row("Type", str(getattr(event, 'type', 'N/A')))
        table.add_row("Timestamp", timestamp)
        
        # Idle Time
        idle_time = getattr(event, 'idle_time', None)
        if idle_time is not None:
            table.add_row("Idle Time", f"{idle_time:.4f}s ({idle_time*1000:.1f}ms)")
        else:
            table.add_row("Idle Time", "N/A")
        
        # Inference Duration
        inference_duration = getattr(event, 'inference_duration_total', None)
        if inference_duration is not None:
            table.add_row("Inference Duration Total", f"{inference_duration:.4f}s ({inference_duration*1000:.1f}ms)")
        else:
            table.add_row("Inference Duration Total", "N/A")
        
        table.add_row("Inference Count", str(getattr(event, 'inference_count', 'N/A')))
        table.add_row("Speech ID", str(getattr(event, 'speech_id', 'N/A')))
        table.add_row("Error", str(getattr(event, 'error', 'None')))
        
        console.print("\n")
        console.print(table)
        console.print("\n")
    
    def increment_active_conversations(self):
        """Increment active conversation counter."""
        if not self.enable_prometheus:
            return
        self.active_conversations.labels(agent_type=self.agent_type).inc()
    
    def decrement_active_conversations(self):
        """Decrement active conversation counter (use in cleanup)."""
        if not self.enable_prometheus:
            return
        self.enqueue_metric_op(
            lambda: self.active_conversations.labels(agent_type=self.agent_type).dec()
        )
    
    def update_config(
        self,
        llm_config: Optional[Dict[str, Any]] = None,
        stt_config: Optional[Dict[str, Any]] = None,
        tts_config: Optional[Dict[str, Any]] = None
    ):
        """
        Update pricing and model configuration dynamically.
        
        Example:
            collector.update_config(
                llm_config={"model": "gpt-4", "price_per_1k_input": 0.03}
            )
        """
        if llm_config:
            self.llm_config.update(llm_config)
            logger.info(f"Updated LLM config: {self.llm_config}")
        
        if stt_config:
            self.stt_config.update(stt_config)
            logger.info(f"Updated STT config: {self.stt_config}")
        
        if tts_config:
            self.tts_config.update(tts_config)
            logger.info(f"Updated TTS config: {self.tts_config}")