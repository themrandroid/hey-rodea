import streamlit as st
import plotly.graph_objects as go
import streamlit.components.v1 as components

def transcript_card(transcript):
    st.markdown(
        """
        <style>
        .transcript-card {
            background-color: #ffffff;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .transcript-title {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
            color: #2E86C1;
        }
        .transcript-text {
            font-size: 15px;
            color: #333;
            line-height: 1.5;
            white-space: pre-wrap;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class='transcript-card'>
            <div class='transcript-text'>{transcript}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def donut_card(label, value, color, key_suffix="", height=220):
    # Create the figure
    fig = go.Figure(data=[go.Pie(
        values=[value, 100 - value],
        hole=0.68,
        marker_colors=[color, "#E8E8E8"],
        textinfo="none",
        hoverinfo="label+percent",
        sort=False
    )])

    fig.update_layout(
        annotations=[
            dict(
                text=f"<b style='color:black;font-size:22px'>{value}%</b><br>"
                     f"<span style='color:#555;font-size:14px'>{label}</span>",
                x=0.5, y=0.5, showarrow=False
            )
        ],
        showlegend=False,
        margin=dict(t=2, b=2, l=2, r=2),
        height=height,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    with st.container():
        st.markdown("<div class='score-card'>", unsafe_allow_html=True)
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False},
            key=f"{label}_donut_{key_suffix}"  # unique ID
        )
        st.markdown("</div>", unsafe_allow_html=True)

def styled_feedback(feedback_list):
    st.markdown(
        """
        <style>
        .feedback-card {
            background-color: #ffffff;
            border-radius: 12px;
            padding: 12px 16px;
            margin-bottom: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            display: flex;
            align-items: center;
        }
        .feedback-text {
            font-size: 15px;
            color: #333;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Collect HTML for all cards first
    cards_html = ""
    for line in feedback_list:
        if any(word in line.lower() for word in ["great", "nice", "good", "well", "balance"]):
            color = "#28B463"  # green
        elif any(word in line.lower() for word in ["fast", "slow", "long", "scarce"]):
            color = "#CA6F1E"  # orange
        else:
            color = "#2E86C1"  # blue

        cards_html += f"""
        <div class='feedback-card'>
            <p class='feedback-text' style='color:{color};'>{line}</p>
        </div>
        """

    # Render all cards at once (avoids Streamlit’s block padding)
    st.markdown(cards_html, unsafe_allow_html=True)