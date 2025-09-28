import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any
import click

from .core.agent import AgenticAI
from .config import Config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@click.group()
@click.pass_context
def cli(ctx):
    """🤖 Agentic AI Research Framework CLI"""
    ctx.ensure_object(dict)
    
    # Initialize configuration
    config = Config()
    ctx.obj['config'] = config
    ctx.obj['ai'] = AgenticAI(config._config)

@cli.command()
@click.argument('query')
@click.option('--no-rag', is_flag=True, help='Disable RAG system')
@click.option('--save', type=str, help='Save result to file')
@click.pass_context
def research(ctx, query: str, no_rag: bool, save: str):
    """🔍 Conduct research on a query"""
    async def _research():
        ai = ctx.obj['ai']
        
        click.echo(f"🔍 Researching: {query}")
        click.echo("=" * 50)
        
        try:
            result = await ai.research(query, use_rag=not no_rag)
            
            click.echo("📋 REASONING STEPS:")
            for i, step in enumerate(result.reasoning_chain, 1):
                click.echo(f"  {i}. {step}")
            
            click.echo("\n📚 SOURCES:")
            for source in result.sources:
                click.echo(f"  • {source}")
            
            click.echo(f"\n💡 ANSWER:")
            click.echo(result.answer)
            
            click.echo(f"\n🎯 Confidence: {result.confidence:.2f}")
            
            if save:
                result_dict = {
                    'query': result.query,
                    'answer': result.answer,
                    'reasoning_chain': result.reasoning_chain,
                    'sources': result.sources,
                    'confidence': result.confidence,
                    'timestamp': result.timestamp.isoformat()
                }
                with open(save, 'w') as f:
                    json.dump(result_dict, f, indent=2)
                click.echo(f"💾 Result saved to {save}")
                
        except Exception as e:
            click.echo(f"❌ Error: {e}")
            logger.exception("Research error")
    
    asyncio.run(_research())

@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--metadata', type=str, help='JSON metadata for the document')
@click.pass_context
def upload(ctx, file_path: str, metadata: str):
    """📤 Upload a document to the knowledge base"""
    async def _upload():
        ai = ctx.obj['ai']
        
        meta_dict = {}
        if metadata:
            try:
                meta_dict = json.loads(metadata)
            except json.JSONDecodeError:
                click.echo("❌ Invalid JSON metadata")
                return
        
        click.echo(f"📤 Uploading: {file_path}")
        
        success = await ai.add_knowledge(file_path, meta_dict)
        
        if success:
            click.echo("✅ Document uploaded successfully")
        else:
            click.echo("❌ Failed to upload document")
    
    asyncio.run(_upload())

@cli.command()
@click.pass_context
def history(ctx):
    """📚 Show research history"""
    ai = ctx.obj['ai']
    
    history = ai.get_research_history()
    
    if not history:
        click.echo("📭 No research history found")
        return
    
    click.echo("📚 RESEARCH HISTORY:")
    click.echo("=" * 50)
    
    for i, result in enumerate(history, 1):
        click.echo(f"{i}. {result.query}")
        click.echo(f"   🕐 {result.timestamp}")
        click.echo(f"   🎯 Confidence: {result.confidence:.2f}")
        click.echo()

@cli.command()
@click.argument('file_path')
@click.pass_context
def save_session(ctx, file_path: str):
    """💾 Save current session"""
    ai = ctx.obj['ai']
    ai.save_session(file_path)
    click.echo(f"💾 Session saved to {file_path}")

@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.pass_context
def load_session(ctx, file_path: str):
    """📂 Load previous session"""
    ai = ctx.obj['ai']
    ai.load_session(file_path)
    click.echo(f"📂 Session loaded from {file_path}")

@cli.command()
@click.pass_context
def interactive(ctx):
    """🤖 Start interactive mode"""
    ai = ctx.obj['ai']
    
    click.echo("🤖 Agentic AI Interactive Mode")
    click.echo("Commands: research <query>, upload <file>, history, save <file>, load <file>, quit")
    click.echo("=" * 50)
    
    while True:
        try:
            command = click.prompt("🔸", type=str).strip()
            
            if command.lower() in ['quit', 'exit', 'q']:
                break
            
            # Parse command
            parts = command.split(' ', 1)
            cmd = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""
            
            if cmd == 'research' and args:
                result = asyncio.run(ai.research(args))
                click.echo(f"💡 {result.answer}")
            
            elif cmd == 'upload' and args:
                success = asyncio.run(ai.add_knowledge(args))
                click.echo("✅ Uploaded" if success else "❌ Failed")
            
            elif cmd == 'history':
                history = ai.get_research_history()
                for i, result in enumerate(history[-5:], 1):  # Show last 5
                    click.echo(f"{i}. {result.query}")
            
            elif cmd == 'save' and args:
                ai.save_session(args)
                click.echo(f"💾 Saved to {args}")
            
            elif cmd == 'load' and args:
                ai.load_session(args)
                click.echo(f"📂 Loaded from {args}")
            
            else:
                click.echo("❓ Unknown command")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            click.echo(f"❌ Error: {e}")
    
    click.echo("👋 Goodbye!")

if __name__ == "__main__":
    cli()