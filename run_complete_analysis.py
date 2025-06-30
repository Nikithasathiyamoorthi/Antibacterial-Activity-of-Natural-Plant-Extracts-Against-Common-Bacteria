import subprocess
import sys
import os

def run_script(script_name, description):
    """Run a Python script and handle errors"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print('='*60)
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=False, 
                              text=True, 
                              cwd=os.path.dirname(os.path.abspath(__file__)))
        
        if result.returncode == 0:
            print(f"✅ {description} completed successfully!")
        else:
            print(f"❌ {description} failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Error running {description}: {str(e)}")
        return False
    
    return True

def main():
    """Run complete antibacterial activity analysis pipeline"""
    print("ANTIBACTERIAL ACTIVITY ANALYSIS - COMPLETE PIPELINE")
    print("="*60)
    print("This script will run the complete analysis pipeline:")
    print("1. Generate synthetic data")
    print("2. Perform statistical analysis") 
    print("3. Create visualizations")
    print("4. Run machine learning analysis")
    print("5. Generate comprehensive report")
    print()
    
    # List of scripts to run in order
    analysis_steps = [
        ("data_generator.py", "Data Generation"),
        ("statistical_analysis.py", "Statistical Analysis"),
        ("data_visualization.py", "Data Visualization"),
        ("machine_learning_analysis.py", "Machine Learning Analysis"),
        ("report_generator.py", "Report Generation")
    ]
    
    # Track success of each step
    results = {}
    
    # Run each analysis step
    for script, description in analysis_steps:
        success = run_script(script, description)
        results[description] = success
        
        if not success:
            print(f"\n⚠️  Warning: {description} failed. Continuing with remaining steps...")
    
    # Final summary
    print(f"\n{'='*60}")
    print("ANALYSIS PIPELINE SUMMARY")
    print('='*60)
    
    for step, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{step:<30} {status}")
    
    successful_steps = sum(results.values())
    total_steps = len(results)
    
    print(f"\nOverall Success Rate: {successful_steps}/{total_steps} ({(successful_steps/total_steps)*100:.1f}%)")
    
    if successful_steps == total_steps:
        print("\n🎉 Complete analysis pipeline executed successfully!")
        print("\nGenerated files:")
        print("📊 Data: antibacterial_data.csv")
        print("📈 Visualizations: *.png files")
        print("📋 HTML Report: antibacterial_analysis_report.html")
        print("📄 Summary: analysis_summary.txt")
        print("\n💡 Open the HTML report in your browser for the complete analysis!")
    else:
        print(f"\n⚠️  Pipeline completed with {total_steps - successful_steps} failed step(s).")
        print("Check the error messages above for troubleshooting.")

if __name__ == "__main__":
    main()
