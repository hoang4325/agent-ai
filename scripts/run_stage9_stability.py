"""
Stage 9 — Stability Campaign Runner (5 RUNS)
==============================================
Executes the full Stage 9 scenario campaign multiple times to verify
deterministic stability.

Run:
  python scripts/run_stage9_stability.py --runs 5
"""
import argparse
import sys
import time
from pathlib import Path
from colorama import init, Fore, Style

# Make sure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.run_stage9_campaign import main as campaign_main


def main():
    init()
    parser = argparse.ArgumentParser(description="Stage 9 Stability Runner")
    parser.add_argument("--runs", type=int, default=5, help="Number of times to run the full campaign")
    args = parser.parse_args()

    print(f"\n{Fore.CYAN}{'═'*70}")
    print(f"  STAGE 9 STABILITY CAMPAIGN  —  {args.runs} RUNS")
    print(f"{'═'*70}{Style.RESET_ALL}\n")

    overall_pass = True
    pass_count = 0
    start_time = time.monotonic()

    # Save original sys.argv
    original_argv = sys.argv.copy()

    for i in range(1, args.runs + 1):
        run_id = f"stage9_stability_run_{i}"
        print(f"{Fore.YELLOW}─── RUN {i} / {args.runs} ─── [ ID: {run_id} ]{Style.RESET_ALL}")
        
        # Override sys.argv for the campaign main parser
        sys.argv = ["run_stage9_campaign.py", "--run-id", run_id]
        
        try:
            # campaign_main returns 0 on PASS, 1 on FAIL
            exit_code = campaign_main()
            if exit_code == 0:
                pass_count += 1
                print(f"{Fore.GREEN}Run {i} SUCCESS{Style.RESET_ALL}\n")
            else:
                overall_pass = False
                print(f"{Fore.RED}Run {i} FAILED{Style.RESET_ALL}\n")
                break
        except Exception as e:
            overall_pass = False
            print(f"{Fore.RED}Run {i} EXCEPTION: {e}{Style.RESET_ALL}\n")
            break

    # Restore sys.argv
    sys.argv = original_argv

    total_time = (time.monotonic() - start_time)
    print(f"\n{Fore.CYAN}{'═'*70}")
    print(f"  STABILITY REPORT")
    print(f"{'═'*70}{Style.RESET_ALL}")
    print(f"  Total runs : {args.runs}")
    print(f"  Passed     : {pass_count}")
    print(f"  Failed     : {args.runs - pass_count}")
    print(f"  Time taken : {total_time:.2f}s")
    print()

    if overall_pass:
        print(f"  {Fore.GREEN}✓ STABILITY VERIFIED: 100% PASS RATE{Style.RESET_ALL}")
    else:
        print(f"  {Fore.RED}✗ STABILITY FAILED{Style.RESET_ALL}")
    
    print(f"{Fore.CYAN}{'═'*70}{Style.RESET_ALL}\n")
    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
