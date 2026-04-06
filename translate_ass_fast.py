#!/usr/bin/env python3
"""
Tradutor de Legendas ASS Ultra-Rápido para RTX 5080 - CORRIGIDO (Modo Temporada)
"""
import asyncio
import re
import json
import hashlib
import sys
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import pysubs2
import ollama
from tqdm import tqdm
import traceback

# Force UTF-8 output to avoid Windows cp1252 crashes when printing emoji/accents.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(
        encoding="utf-8",
        errors="replace",
        line_buffering=True,
        write_through=True,
    )
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(
        encoding="utf-8",
        errors="replace",
        line_buffering=True,
        write_through=True,
    )

# ============================================================
# CONFIGURAÇÕES
# ============================================================
CONFIG = {
    "model": "qwen2.5:14b",
    "max_parallel": 4,
    "batch_size": 15,
    "temperature": 0.2,
    "max_tokens": 2048,
    "timeout": 300,
    "retry_count": 3,
    "enable_cache": True,
    "cache_file": "translation_cache.json",
    "min_text_length": 2,
    "skip_sfx": True,
    "turbo_mode": False,
    "target_language": "Brazilian Portuguese",
    "system_prompt": """Traduza do inglês para português do Brasil. Regras:
- Natural e fluente
- Preserve nomes próprios
- Mantenha tom do personagem
- Sempre traduza o pronome "I" isolado como "Eu"
- Em contrações de primeira pessoa ("I'm", "I've", "I'll", "I'd"), mantenha o sentido em português
    - Retorne APENAS a tradução, sem explicações"""
}


class ModelUnavailableError(RuntimeError):
    """Erro fatal quando o modelo informado não existe no Ollama."""


def _is_model_not_found_error(exc: Exception) -> bool:
    """Detecta erros que indicam modelo inexistente no Ollama."""
    status_code = getattr(exc, "status_code", None)
    message = str(exc).lower()
    if status_code == 404:
        return True
    return "model" in message and "not found" in message


def _ensure_ollama_model_available(model_name: str) -> None:
    """Valida antecipadamente se o modelo solicitado esta disponivel no Ollama."""
    try:
        ollama.show(model_name)
    except Exception as exc:
        if _is_model_not_found_error(exc):
            raise ModelUnavailableError(
                f'❌ Modelo "{model_name}" não está disponível no Ollama. '
                f'Instale antes com: ollama pull {model_name}'
            ) from exc
        raise RuntimeError(f"❌ Não foi possível validar o modelo no Ollama: {exc}") from exc


# ============================================================
# CLASSE TRADUTOR CORRIGIDA
# ============================================================
class FixedASSTranslator:
    """Tradutor ASS com correções de timeout e None values"""
    def __init__(self, config: Dict):
        """Inicializa configuracao, estatisticas, cache e regras de preprocessamento."""
        self.config = config
        self.stats = {"total": 0, "translated": 0, "failed": 0, "cached": 0, "skipped": 0}
        
        # Cache
        self.cache: Dict[str, str] = {}
        if config["enable_cache"]:
            self._load_cache()
            
        # Regex
        self.tag_pattern = re.compile(r'\{[^}]*\}')
        self.skip_patterns = [
            re.compile(r'^[\s♪♫…~\-_\.]+$'),
            re.compile(r'^\([^)]*\)$'),
            re.compile(r'^\[[^\]]*\]$'),
        ]
        self.sfx_words = {"uh", "ah", "oh", "huh", "eh", "woah", "wow", "hey"}

    def _load_cache(self):
        """Carrega traducoes em cache do disco para reduzir chamadas ao modelo."""
        cache_path = Path(self.config["cache_file"])
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
                print(f"📦 Cache carregado: {len(self.cache)} traduções")
            except Exception as e:
                print(f"⚠️ Erro ao carregar cache: {e}")
                self.cache = {}

    def _save_cache(self):
        """Salva o cache atual em disco quando o recurso esta habilitado."""
        if not self.config["enable_cache"]:
            return
        try:
            with open(self.config["cache_file"], 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ Erro ao salvar cache: {e}")

    def _get_text_hash(self, text: str) -> str:
        """Cria hash estavel do texto para chave de cache, incluindo escapes ASS."""
        cache_key_text = text
        if isinstance(text, str) and any(token in text for token in (r"\N", r"\n", r"\h")):
            cache_key_text = "ass_breaks_v2::" + self.protect_ass_breaks_for_model(text)
        return hashlib.md5(cache_key_text.encode('utf-8')).hexdigest()

    def clean_ass_tags(self, text: str) -> str:
        """Remove tags ASS e devolve texto limpo para avaliacao e traducao."""
        text = self.tag_pattern.sub('', text)
        if self.config["turbo_mode"]:
            text = re.sub(r'\\[a-z]+[0-9]*', '', text)
        return text.strip()

    def protect_ass_breaks_for_model(self, text: str) -> str:
        """Substitui escapes ASS por tokens estaveis para preservar formatacao no prompt."""
        if not isinstance(text, str):
            return text
        protected = text.replace(r"\N", "[[[ASS_BR]]]")
        protected = protected.replace(r"\n", "[[[ASS_br]]]")
        protected = protected.replace(r"\h", "[[[ASS_NBSP]]]")
        return protected

    def restore_ass_breaks_from_model(self, text: str) -> str:
        """Restaura os escapes ASS a partir dos tokens apos receber a traducao."""
        if not isinstance(text, str):
            return text
        restored = re.sub(r"\s*\[\[\[ASS_BR\]\]\]\s*", r"\\N", text)
        restored = re.sub(r"\s*\[\[\[ASS_br\]\]\]\s*", r"\\n", restored)
        restored = re.sub(r"\s*\[\[\[ASS_NBSP\]\]\]\s*", r"\\h", restored)
        return restored.strip()

    def normalize_apostrophes(self, text: str) -> str:
        """Padroniza apostrofos para facilitar comparacoes e substituicoes por regex."""
        return text.replace("’", "'").replace("`", "'").replace("´", "'")

    def normalize_first_person_source(self, text: str) -> str:
        """Expande contracoes com I no texto fonte para reduzir ambiguidades de traducao."""
        normalized = self.normalize_apostrophes(text)
        replacements = {
            r"\b[Ii]'m\b": "I am",
            r"\b[Ii]'ve\b": "I have",
            r"\b[Ii]'ll\b": "I will",
            r"\b[Ii]'d\b": "I would",
        }
        for pattern, replacement in replacements.items():
            normalized = re.sub(pattern, replacement, normalized)
        return normalized

    def fix_first_person_translation(self, source_text: str, translated_text: str) -> str:
        """Aplica salvaguardas para manter traducao correta de primeira pessoa."""
        if not isinstance(translated_text, str):
            return translated_text

        source = self.normalize_apostrophes(source_text).strip()
        translated = self.normalize_apostrophes(translated_text).strip()
        if not translated:
            return translated_text

        source_token = re.sub(r"[^\w']", "", source).lower()
        translated_token = re.sub(r"[^\w']", "", translated).lower()

        # Garantia para o caso "I" isolado.
        if source_token == "i" or translated_token == "i":
            return "Eu"

        # Salvaguarda leve para prefixos que escaparem em inglês.
        english_prefix_map = [
            (r"^i['’]m\b", "Eu estou"),
            (r"^i['’]ve\b", "Eu tenho"),
            (r"^i['’]ll\b", "Eu vou"),
            (r"^i['’]d\b", "Eu"),
            (r"^i am\b", "Eu estou"),
            (r"^i have\b", "Eu tenho"),
            (r"^i will\b", "Eu vou"),
            (r"^i would\b", "Eu"),
            (r"^i\b", "Eu"),
        ]
        for pattern, replacement in english_prefix_map:
            if re.match(pattern, translated, flags=re.IGNORECASE):
                return re.sub(pattern, replacement, translated, count=1, flags=re.IGNORECASE)

        return translated_text

    def should_skip_line(self, text: str) -> Tuple[bool, str]:
        """Determina se linha deve ser pulada (simbolos, SFX, muito curta) e o motivo."""
        clean = self.clean_ass_tags(text)
        # Preserve standalone first-person pronoun for translation (e.g., "I" -> "Eu").
        normalized_word = re.sub(r"[^\w']", "", clean).lower()
        if normalized_word == "i":
            return False, ""
        if len(clean) < self.config["min_text_length"]:
            return True, "too_short"
        for pattern in self.skip_patterns:
            if pattern.match(clean):
                return True, "symbols_only"
        if self.config["skip_sfx"]:
            words = clean.lower().split()
            if len(words) <= 2 and all(w in self.sfx_words for w in words):
                return True, "sfx"
        return False, ""

    def reset_stats(self):
        """Reinicia contadores de estatisticas para o processamento do proximo arquivo."""
        self.stats = {"total": 0, "translated": 0, "failed": 0, "cached": 0, "skipped": 0}

    def parse_ass_file(self, input_path: str) -> Tuple[pysubs2.SSAFile, List[Dict]]:
        """Carrega ASS, filtra linhas validas e prepara estrutura para traducao em lote."""
        print(f"📖 Carregando arquivo: {input_path}")
        subs = pysubs2.load(input_path)
        lines_to_translate = []
        text_hash_map = {}
        
        for idx, event in enumerate(subs):
            if not event.text or event.type == "Comment":
                continue
            skip, reason = self.should_skip_line(event.text)
            if skip:
                self.stats["skipped"] += 1
                continue
                
            clean_text = self.clean_ass_tags(event.text)
            if not clean_text:
                self.stats["skipped"] += 1
                continue
                
            text_hash = self._get_text_hash(clean_text)
            cached_translation = None
            if self.config["enable_cache"] and text_hash in self.cache:
                cached_translation = self.cache[text_hash]
                self.stats["cached"] += 1
                
            line_info = {
                "index": idx, "original_event": event, "original_text": event.text,
                "clean_text": clean_text, "text_hash": text_hash,
                "cached_translation": cached_translation, "translated_text": None,
                "style": event.style, "start": event.start, "end": event.end
            }
            
            if cached_translation is None:
                if text_hash not in text_hash_map:
                    text_hash_map[text_hash] = line_info
                    lines_to_translate.append(line_info)
                else:
                    line_info["linked_to"] = text_hash_map[text_hash]["index"]
                    lines_to_translate.append(line_info)
            else:
                line_info["translated_text"] = cached_translation
                lines_to_translate.append(line_info)
                
        self.stats["total"] = len(lines_to_translate)
        unique_lines = len([l for l in lines_to_translate if l.get("cached_translation") is None and "linked_to" not in l])
        print(f"📊 Carregadas {len(lines_to_translate)} linhas totais")
        print(f"   → Linhas únicas: {unique_lines}")
        print(f"   → Em cache: {self.stats['cached']}")
        print(f"   → Puladas: {self.stats['skipped']}")
        return subs, lines_to_translate

    async def translate_single_batch(
        self,
        texts: List[str],
        batch_num: int,
        total_batches: int,
        original_texts: Optional[List[str]] = None,
    ) -> List[str]:
        """Traduz um batch via Ollama com retry, timeout e fallback para texto original."""
        fallback_texts = original_texts if original_texts and len(original_texts) == len(texts) else texts
        numbered_lines = "\n".join([f"{i+1}. {text}" for i, text in enumerate(texts)])
        prompt = (
            f"Traduza cada frase para {self.config['target_language']}. Mantenha a numeração.\n"
            "Mantenha os tokens [[[ASS_BR]]], [[[ASS_br]]] e [[[ASS_NBSP]]] exatamente como estão.\n"
            f"{numbered_lines}\n"
            "Tradução:"
        )
        
        for attempt in range(self.config["retry_count"]):
            try:
                print(f"   🔄 Batch {batch_num}/{total_batches} - Tentativa {attempt+1}...")
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        ollama.generate,
                        model=self.config["model"],
                        prompt=prompt,
                        system=self.config["system_prompt"],
                        options={
                            "temperature": self.config["temperature"],
                            "num_predict": self.config["max_tokens"],
                            "top_p": 0.9,
                        }
                    ),
                    timeout=self.config["timeout"]
                )
                
                translated = []
                response_text = response['response'].strip()
                lines = response_text.split('\n')
                
                for line in lines:
                    line = line.strip()
                    if not line: continue
                    match = re.match(r'^(\d+)[\.\-:\)]\s*(.*)$', line)
                    if match:
                        num = int(match.group(1))
                        text = match.group(2).strip()
                        if 1 <= num <= len(texts):
                            while len(translated) < num:
                                translated.append(None)
                            translated[num-1] = text if text else fallback_texts[num-1]
                    elif len(translated) < len(texts) and not any(c.isdigit() for c in line[:3]):
                        if translated and translated[-1] is not None:
                            translated[-1] += " " + line
                        elif not translated:
                            translated.append(line)
                            
                result = []
                for i in range(len(texts)):
                    if i < len(translated) and translated[i] is not None and translated[i].strip():
                        restored = self.restore_ass_breaks_from_model(translated[i])
                        result.append(restored if restored else fallback_texts[i])
                    else:
                        result.append(fallback_texts[i])
                        
                print(f"   ✅ Batch {batch_num}/{total_batches} concluído!")
                return result
            except asyncio.TimeoutError:
                print(f"   ⏰ Timeout no batch {batch_num} (tentativa {attempt+1})")
                if attempt == self.config["retry_count"] - 1:
                    print(f"   ⚠️ Usando texto original após {self.config['retry_count']} tentativas")
                    return fallback_texts
                await asyncio.sleep(2)
            except Exception as e:
                if _is_model_not_found_error(e):
                    raise ModelUnavailableError(
                        f'❌ Modelo "{self.config["model"]}" não está disponível no Ollama '
                        f"(status code: 404). Encerrando para nova execução com um modelo válido."
                    ) from e
                print(f"   ❌ Erro no batch {batch_num}: {e}")
                if attempt == self.config["retry_count"] - 1:
                    print(f"   ⚠️ Usando texto original")
                    return fallback_texts
                await asyncio.sleep(2)
        return fallback_texts

    async def process_lines_optimized(self, lines: List[Dict]) -> List[Dict]:
        """Processa linhas em lotes, aplica cache e consolida traducoes finais."""
        to_translate = [l for l in lines if l.get("cached_translation") is None and "linked_to" not in l]
        if to_translate:
            batch_size = self.config["batch_size"]
            batches = [to_translate[i:i+batch_size] for i in range(0, len(to_translate), batch_size)]
            total_batches = len(batches)
            print(f"\n📦 Processando {total_batches} batches de até {batch_size} linhas cada")
            
            for batch_num, batch in enumerate(batches, 1):
                original_texts = [item["clean_text"] for item in batch]
                prompt_texts = []
                for text in original_texts:
                    normalized = self.normalize_first_person_source(text)
                    prompt_texts.append(self.protect_ass_breaks_for_model(normalized))
                print(f"\n📦 Batch {batch_num}/{total_batches} - {len(prompt_texts)} linhas")
                if original_texts:
                    preview = original_texts[0][:50] + "..." if len(original_texts[0]) > 50 else original_texts[0]
                    print(f"   Ex: '{preview}'")
                    
                translated_texts = await self.translate_single_batch(
                    prompt_texts,
                    batch_num,
                    total_batches,
                    original_texts=original_texts,
                )
                
                for item, translated in zip(batch, translated_texts):
                    if translated is not None and isinstance(translated, str):
                        fixed_translation = self.fix_first_person_translation(item["clean_text"], translated)
                        item["translated_text"] = fixed_translation
                        if self.config["enable_cache"] and item["text_hash"]:
                            self.cache[item["text_hash"]] = fixed_translation
                        self.stats["translated"] += 1
                    else:
                        item["translated_text"] = item["clean_text"]
                        print(f"   ⚠️ Linha {item['index']} teve tradução inválida, usando original")
                        self.stats["failed"] += 1
                        
                if batch_num % 5 == 0:
                    self._save_cache()
                    print(f"   💾 Cache salvo ({len(self.cache)} entradas)")
                    
        for line in lines:
            if "linked_to" in line:
                original = next((l for l in lines if l["index"] == line["linked_to"]), None)
                if original and original.get("translated_text"):
                    line["translated_text"] = self.fix_first_person_translation(
                        line.get("clean_text", ""),
                        original["translated_text"],
                    )
            elif line.get("cached_translation"):
                fixed_cached = self.fix_first_person_translation(
                    line.get("clean_text", ""),
                    line["cached_translation"],
                )
                line["translated_text"] = fixed_cached
                if (
                    self.config["enable_cache"]
                    and line.get("text_hash")
                    and isinstance(fixed_cached, str)
                    and fixed_cached != line["cached_translation"]
                ):
                    self.cache[line["text_hash"]] = fixed_cached
                
            if line.get("translated_text") is None:
                line["translated_text"] = line.get("clean_text", "")
                print(f"   ⚠️ Linha {line['index']} sem tradução, usando original")
                
        self._save_cache()
        return lines

    def rebuild_ass(self, subs: pysubs2.SSAFile, translated_lines: List[Dict]) -> pysubs2.SSAFile:
        """Reconstrui o ASS final substituindo somente os textos traduzidos mapeados."""
        new_subs = pysubs2.SSAFile()
        new_subs.styles = subs.styles
        new_subs.info = subs.info
        translation_map = {}
        
        for line in translated_lines:
            idx = line["index"]
            translated = line.get("translated_text")
            translation_map[idx] = translated if isinstance(translated, str) else line.get("clean_text", "")
            
        for idx, event in enumerate(subs):
            new_event = pysubs2.SSAEvent()
            new_event.start = event.start
            new_event.end = event.end
            new_event.style = event.style
            new_event.type = event.type
            new_event.marginl = event.marginl
            new_event.marginr = event.marginr
            new_event.marginv = event.marginv
            new_event.effect = event.effect
            
            if idx in translation_map:
                original = event.text
                translated = translation_map[idx]
                clean_original = self.clean_ass_tags(original)
                if clean_original and clean_original in original:
                    new_event.text = original.replace(clean_original, translated, 1)
                else:
                    new_event.text = translated
            else:
                new_event.text = event.text
            new_subs.append(new_event)
        return new_subs

    async def translate_file(self, input_path: str, output_path: Optional[str] = None):
        """Orquestra o fluxo completo de traducao de um episodio e salva a saida."""
        self.reset_stats()
        print(f"\n{'='*60}")
        print(f"🎬 Iniciando tradução CORRIGIDA v2")
        print(f"📁 Arquivo: {Path(input_path).name}")
        print(f"🤖 Modelo: {self.config['model']}")
        print(f"📦 Batch: {self.config['batch_size']} linhas/requisição")
        print(f"⏰ Timeout: {self.config['timeout']}s")
        print(f"🔄 Retries: {self.config['retry_count']}")
        print(f"💾 Cache: {'Ativado' if self.config['enable_cache'] else 'Desativado'}")
        print(f"{'='*60}")
        
        start_time = datetime.now()
        subs, lines = self.parse_ass_file(input_path)
        if not lines:
            print("⚠️ Nenhuma linha para traduzir")
            return
            
        print(f"\n🔄 Processando {len(lines)} linhas...")
        translated_lines = await self.process_lines_optimized(lines)
        
        print("\n🔨 Reconstruindo arquivo ASS...")
        output_subs = self.rebuild_ass(subs, translated_lines)
        
        if output_path is None:
            input_file = Path(input_path)
            output_path = input_file.parent / f"{input_file.stem}_traduzido.ass"
        output_subs.save(str(output_path))
        
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"\n{'='*60}")
        print(f"✅ TRADUÇÃO DO EPISÓDIO CONCLUÍDA!")
        print(f"📄 Saída: {output_path}")
        print(f"📊 Estatísticas:")
        print(f"   • Total linhas: {self.stats['total']}")
        print(f"   • Traduzidas: {self.stats['translated']}")
        print(f"   • Em cache: {self.stats['cached']}")
        print(f"   • Puladas: {self.stats['skipped']}")
        print(f"   • Falhas: {self.stats['failed']}")
        print(f"\n⏱️ Tempo: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        if self.stats['translated'] > 0:
            print(f"🎯 Throughput: {self.stats['translated']/elapsed:.2f} linhas/seg")
        print(f"{'='*60}")


def shutdown_ollama_model(model_name: str):
    """Tenta descarregar da memoria o modelo usado para liberar recursos da maquina."""
    if not model_name:
        return

    print(f"\n🛑 Encerrando modelo Ollama: {model_name}")

    try:
        running = ollama.ps()
        loaded_models = []
        for model_info in getattr(running, "models", []):
            if getattr(model_info, "model", None):
                loaded_models.append(model_info.model)
            if getattr(model_info, "name", None):
                loaded_models.append(model_info.name)

        if model_name not in loaded_models:
            print(f"ℹ️ Modelo '{model_name}' já não está carregado.")
            return
    except Exception as e:
        print(f"⚠️ Não foi possível consultar modelos ativos ({e}). Tentando encerrar mesmo assim...")

    try:
        ollama.generate(model=model_name, prompt="", keep_alive=0)
        print(f"✅ Modelo '{model_name}' descarregado via API.")
        return
    except Exception as e:
        print(f"⚠️ Falha ao descarregar via API: {e}")

    ollama_cli = shutil.which("ollama")
    if not ollama_cli:
        print("⚠️ Comando 'ollama' não encontrado para fallback de encerramento.")
        return

    try:
        result = subprocess.run(
            [ollama_cli, "stop", model_name],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        details = (result.stdout or result.stderr).strip()
        if result.returncode == 0:
            print(f"✅ Modelo '{model_name}' encerrado via CLI.")
            if details:
                print(f"   ↳ {details}")
        else:
            print(f"⚠️ Não foi possível encerrar o modelo via CLI (código {result.returncode}).")
            if details:
                print(f"   ↳ {details}")
    except Exception as e:
        print(f"⚠️ Erro ao tentar encerrar via CLI: {e}")

# ============================================================
# FUNÇÃO PRINCIPAL (MODO TEMPORADA)
# ============================================================
async def main():
    """Entrada principal em modo temporada: valida argumentos e processa todos os .ass."""
    import argparse
    parser = argparse.ArgumentParser(description="Tradutor ASS em Lote para Temporadas (RTX 5080)")
    parser.add_argument("-i", "--input-dir", default="./entrada", help="Pasta com arquivos .ass de entrada")
    parser.add_argument("-o", "--output-dir", default="./saida", help="Pasta de saída para arquivos traduzidos")
    parser.add_argument("-m", "--model", default=CONFIG["model"], help="Modelo Ollama")
    parser.add_argument("--batch-size", type=int, default=CONFIG["batch_size"], help="Batch size")
    parser.add_argument("--timeout", type=int, default=CONFIG["timeout"], help="Timeout em segundos")
    parser.add_argument("--turbo", action="store_true", help="Modo turbo")
    parser.add_argument("--clear-cache", action="store_true", help="Limpa cache antes de iniciar")
    parser.add_argument("--no-cache", action="store_true", help="Desativa cache")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"❌ Pasta de entrada não encontrada: {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    ass_files = sorted(input_dir.glob("*.ass"))
    
    if not ass_files:
        print(f"⚠️ Nenhum arquivo .ass encontrado em {input_dir}")
        return

    print(f"📂 Encontrados {len(ass_files)} arquivos .ass para tradução.")
    print(f"📤 Saída: {output_dir.resolve()}")

    if args.clear_cache:
        cache_file = Path(CONFIG["cache_file"])
        if cache_file.exists():
            cache_file.unlink()
        print("🗑️ Cache limpo!")

    config = CONFIG.copy()
    config["model"] = args.model
    config["batch_size"] = args.batch_size
    config["timeout"] = args.timeout
    config["turbo_mode"] = args.turbo
    config["enable_cache"] = not args.no_cache
    if args.turbo:
        config["temperature"] = 0.15
        config["batch_size"] = min(config["batch_size"], 10)
        print("🔥 MODO TURBO ATIVADO")

    try:
        _ensure_ollama_model_available(config["model"])
    except ModelUnavailableError as exc:
        print(str(exc))
        print("⛔ Encerrando backend para nova tentativa com modelo correto.")
        raise SystemExit(2)
    except Exception as exc:
        print(f"❌ Erro ao validar modelo no Ollama: {exc}")
        raise SystemExit(2)

    translator = FixedASSTranslator(config)
    season_start = datetime.now()
    processed_count = 0
    failed_files = []

    try:
        for ass_file in tqdm(ass_files, desc="📦 Processando Temporada", unit="ep"):
            print(f"\n{'='*60}")
            print(f"🎬 INICIANDO: {ass_file.name}")
            print(f"{'='*60}")
            
            output_file = output_dir / f"{ass_file.stem}.pt.ass"
            try:
                await translator.translate_file(str(ass_file), str(output_file))
                processed_count += 1
            except ModelUnavailableError as e:
                print(str(e))
                print("⛔ Encerrando backend para nova tentativa com modelo correto.")
                raise SystemExit(2)
            except Exception as e:
                print(f"❌ ERRO CRÍTICO ao processar {ass_file.name}: {e}")
                traceback.print_exc()
                failed_files.append(ass_file.name)
                
            translator._save_cache()

        season_elapsed = (datetime.now() - season_start).total_seconds()
        print(f"\n{'='*60}")
        print(f"✅ TRADUÇÃO DA TEMPORADA CONCLUÍDA!")
        print(f"📊 Arquivos processados: {processed_count}/{len(ass_files)}")
        if failed_files:
            print(f"⚠️ Falhas: {', '.join(failed_files)}")
        print(f"💾 Cache final: {len(translator.cache)} entradas")
        print(f"⏱️ Tempo total da temporada: {season_elapsed:.1f}s ({season_elapsed/60:.1f} min)")
        print(f"{'='*60}")
    finally:
        shutdown_ollama_model(config["model"])

if __name__ == "__main__":
    asyncio.run(main())
