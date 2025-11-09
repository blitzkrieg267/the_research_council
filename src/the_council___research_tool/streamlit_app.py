import streamlit as st
from pathlib import Path
import tempfile
import os
from datetime import datetime
from intelligent_analysis_agent import IntelligentAnalysisAgent

def main():
    st.title("🧠 Intelligent Document Analysis Agent")
    st.markdown("Upload documents and get comprehensive AI analysis with iterative critique and improvement using Azure OpenAI.")

    # Initialize the intelligent analysis agent
    analysis_agent = IntelligentAnalysisAgent()

    # Sidebar for options
    with st.sidebar:
        st.header("⚙️ Analysis Options")
        analysis_mode = st.radio(
            "Analysis Mode:",
            ["Quick Summary", "Council Analysis (Live Reasoning)", "External Research Council"],
            help="Quick Summary: Basic AI summarization\nCouncil Analysis: Multi-agent critique with live reasoning display\nExternal Research: Full research intelligence with arXiv/Wikipedia"
        )

        max_iterations = st.slider(
            "Critique Iterations",
            min_value=1,
            max_value=5,
            value=3,
            disabled=(analysis_mode == "Quick Summary")
        )

        # Council research options
        if analysis_mode == "External Research Council":
            st.subheader("🏛️ Council Configuration")
            council_topic = st.text_input(
                "Research Topic:",
                placeholder="e.g., renewable energy efficiency, quantum computing applications",
                help="Topic for comprehensive multi-agent research analysis"
            )
            council_iterations = st.slider(
                "Council Iterations:",
                min_value=1,
                max_value=5,
                value=3,
                help="Number of critique iterations in the council session"
            )

            # Tavily API Configuration
            st.subheader("🔍 Web Research Configuration")
            st.success("✅ Tavily API key configured automatically!")
            st.info("Tavily will search the web, extract content, and provide AI-powered answers for comprehensive research.")
            st.markdown("**Features enabled:**")
            st.markdown("- 🔍 Advanced web search with AI-powered answers")
            st.markdown("- 📄 Content extraction from top sources")
            st.markdown("- 🧠 Vector embeddings for retrieved content")
            st.markdown("- 📝 Automated content summarization")

        st.markdown("---")
        st.markdown("### 📚 Document Library")
        try:
            documents = analysis_agent.document_agent.get_all_documents()
            if documents:
                st.success(f"📄 {len(documents)} documents stored")
                for doc in documents[:5]:  # Show first 5
                    st.text(f"• {doc}")
                if len(documents) > 5:
                    st.text(f"... and {len(documents) - 5} more")
            else:
                st.info("No documents processed yet")
        except:
            st.info("Document library not initialized")

    # Main content
    if analysis_mode == "External Research Council":
        # Council Research Interface
        st.subheader("🏛️ The Council - Multi-Agent Research Intelligence")

        if 'council_topic' in locals() and council_topic.strip():
            st.success(f"🎯 Research Topic: **{council_topic}**")

            # Council research button
            if st.button(f"🏛️ Convene The Council ({council_iterations} iterations)", type="primary"):
                council_placeholder = st.empty()

                try:
                    from the_council import TheCouncil
                    council = TheCouncil()

                    # Display live updates
                    with council_placeholder.container():
                        st.subheader("🔴 LIVE: Council Session in Progress")
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        live_log = st.empty()

                        updates_received = 0
                        total_updates = 10  # Estimated

                        # Collect all updates
                        all_updates = []
                        final_result_data = None

                        for update in council.conduct_enhanced_research_council(council_topic, max_iterations=council_iterations):
                            all_updates.append(update)
                            updates_received += 1

                            # Update progress
                            progress = min(updates_received / total_updates, 0.9)
                            progress_bar.progress(progress)

                            # Update status
                            agent = update['agent']
                            msg_type = update['message_type']
                            content = update['content']

                            if msg_type == 'session_start':
                                status_text.text(f"🎯 {content}")
                            elif msg_type == 'phase_start':
                                status_text.text(f"📋 {content}")
                            elif msg_type == 'iteration_start':
                                status_text.text(f"🔄 {content}")
                            elif msg_type == 'action':
                                status_text.text(f"🤖 {agent}: {content}")
                            elif msg_type == 'result':
                                status_text.text(f"✅ {content}")
                            elif msg_type == 'session_complete':
                                status_text.text(f"🎉 {content}")
                                progress_bar.progress(1.0)
                            elif msg_type == 'final_result':
                                final_result_data = content
                                progress_bar.progress(1.0)

                        # Display final results
                        st.success("🏛️ Council Session Completed!")

                        # Get final results
                        session_summary = council.get_session_summary()

                        # Results tabs
                        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                            "📊 Collective Insight Report",
                            "🔍 Research Analysis",
                            "🔄 Critique History",
                            "📋 Session Log",
                            "🤖 Agent Efforts",
                            "📈 Session Summary"
                        ])

                        with tab1:
                            st.subheader("🎯 Collective Insight Report")
                            # Get the final synthesis result
                            if final_result_data and 'collective_insight_report' in final_result_data:
                                insight_report = final_result_data['collective_insight_report']
                                if 'collective_insight_report' in insight_report:
                                    st.markdown(insight_report['collective_insight_report'])
                                else:
                                    st.info("Collective Insight Report completed but content not available.")
                            else:
                                st.info("Collective Insight Report will be displayed here upon completion.")

                        with tab2:
                            st.subheader("🔍 Research Analysis")
                            if final_result_data and 'research_analysis' in final_result_data:
                                research = final_result_data['research_analysis']
                                if 'analysis' in research:
                                    st.markdown("### Research Findings")
                                    st.markdown(research['analysis'])

                                    if 'sources' in research and research['sources']:
                                        st.markdown("### Sources Analyzed")
                                        for source in research['sources'][:5]:  # Show first 5
                                            st.markdown(f"- **{source.get('title', 'Unknown')}** ({source.get('source', 'Unknown')})")
                                else:
                                    st.info("Research analysis completed but content not available.")
                            else:
                                st.info("Research findings from arXiv and Wikipedia will be displayed here.")

                        with tab3:
                            st.subheader("🔄 Critique & Improvement Process")
                            if final_result_data and 'critique_history' in final_result_data:
                                critique_history = final_result_data['critique_history']
                                if critique_history:
                                    for i, critique in enumerate(critique_history, 1):
                                        with st.expander(f"📝 Iteration {i} Critique & Improvement", expanded=(i == len(critique_history))):
                                            if 'critique' in critique:
                                                st.markdown("**Critique Feedback:**")
                                                st.markdown(critique['critique'][:1000] + "..." if len(critique.get('critique', '')) > 1000 else critique.get('critique', ''))
                                            if 'improved_summary' in critique:
                                                st.markdown("**Improved Summary:**")
                                                st.markdown(critique['improved_summary'][:1000] + "..." if len(critique.get('improved_summary', '')) > 1000 else critique.get('improved_summary', ''))
                                else:
                                    st.info("No critique iterations were completed.")
                            else:
                                st.info("Iterative critique history will be displayed here.")

                        with tab4:
                            st.subheader("📋 Session Conversation Log")
                            # Display conversation log
                            for update in all_updates[-20:]:  # Show last 20 updates
                                agent = update['agent']
                                msg_type = update['message_type']
                                content = update['content']
                                timestamp = update.get('timestamp', '')[:19]  # Format timestamp

                                with st.expander(f"{timestamp} - {agent}: {msg_type}", expanded=False):
                                    st.write(content)
                                    if update.get('metadata'):
                                        st.json(update['metadata'])

                        with tab5:
                            st.subheader("🤖 Agent Efforts & Detailed Logs")
                            st.markdown("**Real-time tracking of each agent's work, processing time, and decision-making process**")

                            # Extract effort details from final results
                            if final_result_data:
                                # Look for effort details in the iterative results
                                iterative_results = final_result_data.get('iterative_qa_results', [])

                                if iterative_results:
                                    for iteration_idx, iteration_data in enumerate(iterative_results):
                                        iteration_num = iteration_idx + 1
                                        with st.expander(f"🔄 Iteration {iteration_num} - Agent Efforts", expanded=(iteration_num == len(iterative_results))):

                                            # Research Agent Efforts
                                            research_qa = iteration_data.get('research_qa', {})
                                            effort_log = research_qa.get('effort_log', {})

                                            if effort_log:
                                                st.markdown("#### 🧠 Research Agent Efforts")
                                                col1, col2, col3 = st.columns(3)
                                                with col1:
                                                    st.metric("Questions Answered", effort_log.get('question_analysis', {}).get('questions_processed', 0))
                                                with col2:
                                                    st.metric("Successful Answers", effort_log.get('question_analysis', {}).get('successful_answers', 0))
                                                with col3:
                                                    st.metric("Processing Time", f"{effort_log.get('total_duration_seconds', 0):.1f}s")

                                                # Show question-by-question breakdown
                                                question_details = effort_log.get('question_analysis', {}).get('question_details', [])
                                                if question_details:
                                                    st.markdown("**Question Processing Details:**")
                                                    for q_detail in question_details:
                                                        with st.container():
                                                            q_num = q_detail.get('question_number', '?')
                                                            q_text = q_detail.get('question_text', '')[:50] + '...'
                                                            status = q_detail.get('status', 'unknown')
                                                            duration = q_detail.get('duration_seconds', 0)

                                                            status_icon = "✅" if status == "completed" else "❌" if status == "failed" else "⏳"
                                                            st.markdown(f"{status_icon} **Q{q_num}**: {q_text} ({duration:.1f}s)")

                                            # Critique Agent Efforts
                                            critique_qa = iteration_data.get('critique_qa', {})
                                            if critique_qa.get('effort_log'):
                                                crit_effort = critique_qa.get('effort_log', {})
                                                st.markdown("#### 🧐 Critique Agent Efforts")
                                                col1, col2 = st.columns(2)
                                                with col1:
                                                    st.metric("Critiques Provided", crit_effort.get('total_critiques', 0))
                                                with col2:
                                                    st.metric("Processing Time", f"{crit_effort.get('total_duration_seconds', 0):.1f}s")

                                            # Reviewer Agent Efforts
                                            review_qa = iteration_data.get('review_qa', {})
                                            if review_qa.get('effort_log'):
                                                rev_effort = review_qa.get('effort_log', {})
                                                st.markdown("#### 👨‍⚖️ Reviewer Agent Efforts")
                                                col1, col2 = st.columns(2)
                                                with col1:
                                                    st.metric("Reviews Completed", rev_effort.get('total_reviews', 0))
                                                with col2:
                                                    st.metric("Processing Time", f"{rev_effort.get('total_duration_seconds', 0):.1f}s")
                                else:
                                    st.info("No iterative analysis data available for effort tracking.")

                                # Query Refinement Efforts
                                query_refinement = final_result_data.get('query_refinement', {})
                                if query_refinement.get('effort_log'):
                                    qr_effort = query_refinement.get('effort_log', {})
                                    st.markdown("### 🔍 Query Refinement Agent")
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Queries Generated", qr_effort.get('queries_generated', 0))
                                    with col2:
                                        st.metric("Parsing Success", "✅" if qr_effort.get('parsing_successful', False) else "❌")
                                    with col3:
                                        st.metric("Processing Time", f"{qr_effort.get('total_duration_seconds', 0):.1f}s")

                                # Tavily Research Efforts
                                enhanced_research = final_result_data.get('enhanced_research', {})
                                if enhanced_research.get('effort_details'):
                                    tav_effort = enhanced_research.get('effort_details', {})
                                    st.markdown("### 🔍 Tavily Research Agent")
                                    stages = tav_effort.get('stages', [])
                                    if stages:
                                        for stage in stages:
                                            stage_name = stage.get('stage', 'Unknown').replace('_', ' ').title()
                                            duration = stage.get('duration_seconds', 0)
                                            st.metric(f"{stage_name} Duration", f"{duration:.1f}s")

                                            if stage.get('stage') == 'web_search':
                                                queries_processed = stage.get('queries_processed', 0)
                                                st.metric("Queries Processed", queries_processed)

                                            elif stage.get('stage') == 'content_processing':
                                                embeddings = stage.get('embedding_batches', 0)
                                                summaries = stage.get('summaries_created', 0)
                                                st.metric("Embeddings Created", embeddings)
                                                st.metric("Summaries Created", summaries)

                                    # Overall workflow stats
                                    total_duration = tav_effort.get('total_duration_seconds', 0)
                                    sources_found = tav_effort.get('total_sources_found', 0)
                                    embeddings_created = tav_effort.get('total_embeddings_created', 0)

                                    st.markdown("**Overall Workflow Stats:**")
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Total Duration", f"{total_duration:.1f}s")
                                    with col2:
                                        st.metric("Sources Found", sources_found)
                                    with col3:
                                        st.metric("Embeddings Created", embeddings_created)

                            else:
                                st.info("No effort tracking data available. This feature requires the latest agent implementations.")

                        with tab6:
                            st.subheader("📈 Session Summary")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Total Interactions", session_summary['total_interactions'])
                                st.metric("Agents Involved", len(session_summary['agents_involved']))
                            with col2:
                                duration = session_summary.get('end_time', datetime.now()) - session_summary.get('start_time', datetime.now())
                                st.metric("Session Duration", f"{duration.seconds}s")
                                st.metric("Status", session_summary['status'].upper())

                            st.markdown("**Agents Involved:**")
                            for agent in session_summary['agents_involved']:
                                st.markdown(f"- 🤖 {agent.replace('_', ' ').title()}")

                except Exception as e:
                    st.error(f"❌ Council session failed: {str(e)}")
                    st.info("💡 Try a different research topic or check your network connection.")

        else:
            st.warning("⚠️ Please enter a research topic in the sidebar to begin a Council Research Session.")
            st.info("💡 Example topics: 'renewable energy efficiency', 'quantum computing applications', 'machine learning ethics'")

    else:
        # Document Analysis Interface
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("📤 Document Upload")

            # File uploader with expanded file types
            uploaded_file = st.file_uploader(
                "Choose a document",
                type=['txt', 'md', 'py', 'js', 'html', 'css', 'json', 'xml', 'pdf', 'docx', 'xlsx', 'xls'],
                help="Upload documents for AI analysis (PDF, Word, Excel, and text files supported)"
            )

        with col2:
            st.subheader("🎯 Analysis Type")
            if analysis_mode == "Quick Summary":
                st.info("⚡ **Quick Summary**\n- Basic AI summarization\n- Fast processing\n- Single output")
            elif analysis_mode == "Council Analysis (Live Reasoning)":
                st.info(f"🏛️ **Council Analysis**\n- Live agent interactions\n- {max_iterations} critique iterations\n- Comprehensive insights")

        if uploaded_file is not None:
            # Display file info
            st.success(f"📄 File uploaded: {uploaded_file.name}")
            st.info(f"📏 File size: {len(uploaded_file.getvalue()):,} bytes")

            # File type detection
            file_extension = Path(uploaded_file.name).suffix.lower()
            if file_extension in ['.pdf']:
                st.info("📕 PDF document detected - will be processed with advanced parsing")
            elif file_extension in ['.docx']:
                st.info("📝 Word document detected - will extract text content")
            elif file_extension in ['.xlsx', '.xls']:
                st.info("📊 Excel file detected - will process tabular data")
            else:
                st.info("📄 Text file detected - will process directly")

            # Show file preview for text files
            if file_extension in ['.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.xml']:
                file_content = uploaded_file.getvalue().decode('utf-8', errors='ignore')
                with st.expander("👁️ Preview file content"):
                    st.text_area(
                        "Content Preview",
                        file_content[:2000] + ("..." if len(file_content) > 2000 else ""),
                        height=200,
                        disabled=True
                    )

            # Analysis button
            button_text = "📝 Generate Summary" if analysis_mode == "Quick Summary" else f"🏛️ Start Council Analysis ({max_iterations} iterations)"
            if st.button(button_text, type="primary"):
                with st.spinner(f"{'Generating summary' if analysis_mode == 'Quick Summary' else f'Running intelligent analysis with {max_iterations} iterations'}..."):
                    try:
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                            temp_file.write(uploaded_file.getvalue())
                            temp_file_path = temp_file.name

                        if analysis_mode == "Quick Summary":
                            # Use simple summarizer
                            from summarizer import FileSummarizer
                            summarizer = FileSummarizer()
                            summary = summarizer.summarize_file(temp_file_path)

                            # Display results
                            st.success("✅ Summary generated successfully!")
                            st.subheader("📋 Summary")
                            st.markdown(summary)

                            # Download button
                            st.download_button(
                                label="📥 Download Summary",
                                data=summary,
                                file_name=f"summary_{uploaded_file.name}.txt",
                                mime="text/plain"
                            )

                        else:
                            # Use Council Analysis with live reasoning display
                            council_placeholder = st.empty()

                            try:
                                from the_council import TheCouncil
                                council = TheCouncil()

                                # Display live updates
                                with council_placeholder.container():
                                    st.subheader("🔴 LIVE: Council Analysis in Progress")
                                    progress_bar = st.progress(0)
                                    status_text = st.empty()

                                    updates_received = 0
                                    total_updates = 15  # Estimated for document analysis

                                    # Collect all updates
                                    all_updates = []
                                    final_result_data = None

                                    for update in council.conduct_document_council(temp_file_path, max_iterations=max_iterations):
                                        all_updates.append(update)
                                        updates_received += 1

                                        # Update progress
                                        progress = min(updates_received / total_updates, 0.9)
                                        progress_bar.progress(progress)

                                        # Update status
                                        agent = update['agent']
                                        msg_type = update['message_type']
                                        content = update['content']

                                        if msg_type == 'session_start':
                                            status_text.text(f"🎯 {content}")
                                        elif msg_type == 'phase_start':
                                            status_text.text(f"📋 {content}")
                                        elif msg_type == 'iteration_start':
                                            status_text.text(f"🔄 {content}")
                                        elif msg_type == 'action':
                                            status_text.text(f"🤖 {agent}: {content}")
                                        elif msg_type == 'result':
                                            status_text.text(f"✅ {content}")
                                        elif msg_type == 'session_complete':
                                            status_text.text(f"🎉 {content}")
                                            progress_bar.progress(1.0)
                                        elif msg_type == 'final_result':
                                            final_result_data = content
                                            progress_bar.progress(1.0)

                                # Display final results
                                st.success("🏛️ Council Analysis Completed!")

                                # Get final results
                                session_summary = council.get_session_summary()

                                # Results tabs
                                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                                    "📊 Collective Insights",
                                    "📋 Analysis Evolution",
                                    "🔄 Critique Iterations",
                                    "📋 Live Session Log",
                                    "🤖 Agent Efforts",
                                    "📈 Analysis Summary"
                                ])

                                with tab1:
                                    st.subheader("🎯 Collective Insights")
                                    if final_result_data and 'collective_insight_report' in final_result_data:
                                        insight_report = final_result_data['collective_insight_report']
                                        if 'collective_insight_report' in insight_report:
                                            st.markdown(insight_report['collective_insight_report'])
                                        else:
                                            st.info("Collective insights generated but content not available.")
                                    else:
                                        st.info("Collective insights will be displayed here upon completion.")

                                with tab2:
                                    st.subheader("📋 Analysis Evolution")
                                    if final_result_data:
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.markdown("### Initial Summary")
                                            initial = final_result_data.get('initial_summary', 'Not available')
                                            st.markdown(initial[:500] + "..." if len(initial) > 500 else initial)

                                        with col2:
                                            st.markdown("### Final Analysis")
                                            final_analysis = final_result_data.get('final_analysis', 'Not available')
                                            st.markdown(final_analysis[:500] + "..." if len(final_analysis) > 500 else final_analysis)

                                with tab3:
                                    st.subheader("🔄 Critique Iterations")
                                    if final_result_data and 'critique_history' in final_result_data:
                                        critique_history = final_result_data['critique_history']
                                        if critique_history:
                                            for i, critique in enumerate(critique_history, 1):
                                                with st.expander(f"🔄 Iteration {i}: Critique & Refinement", expanded=(i == len(critique_history))):
                                                    if 'critique' in critique:
                                                        st.markdown("**🤔 Critique Feedback:**")
                                                        st.markdown(critique['critique'][:800] + "..." if len(critique.get('critique', '')) > 800 else critique.get('critique', ''))
                                                    if 'improved_summary' in critique:
                                                        st.markdown("**✨ Improved Analysis:**")
                                                        st.markdown(critique['improved_summary'][:800] + "..." if len(critique.get('improved_summary', '')) > 800 else critique.get('improved_summary', ''))
                                        else:
                                            st.info("No critique iterations were completed.")
                                    else:
                                        st.info("Critique history will be displayed here.")

                                with tab4:
                                    st.subheader("📋 Live Session Conversation Log")
                                    # Display conversation log
                                    for update in all_updates[-25:]:  # Show last 25 updates
                                        agent = update['agent']
                                        msg_type = update['message_type']
                                        content = update['content']
                                        timestamp = update.get('timestamp', '')[:19]  # Format timestamp

                                        with st.expander(f"{timestamp} - {agent}: {msg_type}", expanded=False):
                                            st.write(content)
                                            if update.get('metadata'):
                                                st.json(update['metadata'])

                                with tab5:
                                    st.subheader("🤖 Agent Efforts & Detailed Logs")
                                    st.markdown("**Real-time tracking of each agent's work, processing time, and decision-making process**")

                                    if final_result_data:
                                        # Document analysis effort tracking (simpler than research council)
                                        st.markdown("### 📄 Document Processing Agents")

                                        # Summarizer Agent (initial analysis)
                                        if final_result_data.get('initial_summary'):
                                            st.markdown("#### 📝 Summarizer Agent")
                                            summary_length = len(final_result_data.get('initial_summary', ''))
                                            st.metric("Initial Summary Length", f"{summary_length} chars")
                                            st.info("✅ Initial document summary generated")

                                        # Critique iterations
                                        critique_history = final_result_data.get('critique_history', [])
                                        if critique_history:
                                            st.markdown("#### 🧐 Critique Agent")
                                            st.metric("Critique Iterations", len(critique_history))

                                            for i, critique in enumerate(critique_history, 1):
                                                with st.expander(f"Critique Iteration {i}", expanded=(i == len(critique_history))):
                                                    if 'critique' in critique:
                                                        crit_length = len(critique.get('critique', ''))
                                                        st.metric(f"Iteration {i} Critique Length", f"{crit_length} chars")
                                                    if 'improved_summary' in critique:
                                                        imp_length = len(critique.get('improved_summary', ''))
                                                        st.metric(f"Iteration {i} Improved Summary", f"{imp_length} chars")

                                        # Reviewer Agent (final synthesis)
                                        if final_result_data.get('collective_insight_report'):
                                            st.markdown("#### 👨‍⚖️ Reviewer Agent")
                                            insight_report = final_result_data.get('collective_insight_report', {})
                                            if 'collective_insight_report' in insight_report:
                                                report_length = len(insight_report['collective_insight_report'])
                                                st.metric("Final Report Length", f"{report_length} chars")
                                                st.info("✅ Collective insights report generated")

                                        # Document ingestion stats
                                        doc_info = final_result_data.get('document_info', {})
                                        if doc_info:
                                            st.markdown("### 📊 Document Processing Stats")
                                            col1, col2 = st.columns(2)
                                            with col1:
                                                st.metric("File Size", f"{doc_info.get('file_size', 0):,} bytes")
                                                st.metric("Content Extracted", f"{doc_info.get('content_length', 0):,} chars")
                                            with col2:
                                                st.metric("File Type", doc_info.get('file_type', 'Unknown').upper())
                                                st.metric("Processing Status", "✅ Completed")

                                    else:
                                        st.info("No effort tracking data available for document analysis.")

                                with tab6:
                                    st.subheader("📈 Analysis Summary")
                                    if final_result_data:
                                        # Show session info
                                        session_info = final_result_data.get('session_info', {})
                                        doc_info = final_result_data.get('document_info', {})

                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.metric("Document Type", doc_info.get('file_type', 'Unknown').upper())
                                            st.metric("Content Length", f"{doc_info.get('content_length', 0):,} chars")
                                            st.metric("Chunks Processed", doc_info.get('chunks_processed', 0))
                                            st.metric("Iterations Completed", final_result_data.get('iterations_completed', 0))

                                        with col2:
                                            # Calculate duration from session info
                                            start_time = session_info.get('start_time')
                                            end_time = session_info.get('end_time')
                                            if start_time and end_time:
                                                try:
                                                    # Parse ISO format datetime strings
                                                    start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                                                    end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                                                    duration = end_dt - start_dt
                                                    st.metric("Analysis Duration", f"{duration.seconds}s")
                                                except:
                                                    st.metric("Analysis Duration", "Unknown")
                                            else:
                                                st.metric("Analysis Duration", "Unknown")

                                            agents_involved = session_info.get('agents_involved', [])
                                            st.metric("Agents Active", len(agents_involved))

                                        st.markdown("**🤖 Agents Involved:**")
                                        for agent in agents_involved:
                                            st.markdown(f"- {agent.replace('_', ' ').title()}")

                                        # Show additional session stats
                                        st.markdown("**📊 Session Statistics:**")
                                        st.markdown(f"- **Session ID**: {session_info.get('session_id', 'Unknown')}")
                                        st.markdown(f"- **Status**: {session_info.get('status', 'Unknown').upper()}")
                                        st.markdown(f"- **Total Interactions**: {len(final_result_data.get('conversation_log', []))}")
                                    else:
                                        st.error("No analysis data available. The council session may not have completed successfully.")

                            except Exception as e:
                                st.error(f"❌ Council analysis failed: {str(e)}")
                                st.info("💡 Try uploading a smaller file or check the file format.")

                        # Clean up temp file
                        os.unlink(temp_file_path)

                    except Exception as e:
                        st.error(f"❌ Error during analysis: {str(e)}")
                        st.info("💡 Try uploading a smaller file or check the file format.")

    # Instructions
    with st.expander("ℹ️ How to use"):
        st.markdown("""
        ### 📤 Upload Documents
        - **Supported formats**: PDF, Word (.docx), Excel (.xlsx/.xls), and text files
        - **File size limit**: Large files will be processed in chunks

        ### 🎯 Analysis Modes

        **Quick Summary**:
        - Fast AI-powered summarization
        - Single-pass analysis
        - Basic document overview

        **🏛️ Council Analysis (Live Reasoning)**:
        - Multi-agent document analysis with live agent interactions
        - Document ingestion, parsing, and chunking
        - Iterative critique and improvement (configurable iterations)
        - Live display of agent reasoning and communication
        - Comprehensive insights with reasoning traces
        - Document storage for future reference

        **🏛️ External Research Council**:
        - Multi-agent research intelligence system with Tavily web search
        - Advanced web search, content extraction, and AI-powered answers
        - Live agent interaction display with real-time reasoning
        - Collective insight reports with comprehensive analysis
        - No document upload required - just enter a research topic
        - Tavily API key configured automatically (no manual setup required)

        ### 🏛️ Council Analysis Process (For Documents)
        1. **Document Ingestion**: Parser agent loads and chunks documents, stores in vector database
        2. **Initial Analysis**: Summarizer agent creates first-pass comprehensive summary
        3. **Live Critique Loop**: Critique agent reviews and suggests improvements (configurable iterations)
        4. **Comprehensive Review**: Reviewer agent provides final evaluation and synthesis
        5. **Collective Insights**: Final comprehensive report with reasoning traces and agent contributions

        ### 🏛️ External Research Council Process
        1. **Research Gathering**: Agents search arXiv and Wikipedia for relevant papers
        2. **Analysis Phase**: Research agent synthesizes findings from multiple sources
        3. **Critique Iterations**: Critique agent reviews and improves analysis (configurable)
        4. **Senior Review**: Reviewer agent provides comprehensive evaluation
        5. **Synthesis**: Final collective insight report with citations and reasoning traces

        ### 💾 Document Library
        Processed documents are stored locally for future reference and cross-document analysis.
        """)

    # Footer
    st.markdown("---")
    st.markdown("🧠 Built with Azure OpenAI, LangChain, ChromaDB, and Streamlit")

if __name__ == "__main__":
    main()
