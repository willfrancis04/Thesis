"""Shared helpers for thesis box-and-whisker figures."""


def add_figure_n_key(fig, text='n = number of swings', fontsize=9, y_fig=0.018):
    """
    Single note at the bottom-right of the figure (figure coordinates, outside axes).
    y_fig: vertical position in figure coords (0=bottom); raise slightly if trimming bottom margin.
    Call after layout (tight_layout / subplots_adjust) and before savefig.
    Returns the Text artist (pass in bbox_extra_artists with bbox_inches='tight' so it is not clipped).
    """
    return fig.text(
        0.99,
        y_fig,
        text,
        ha='right',
        va='bottom',
        fontsize=fontsize,
        transform=fig.transFigure,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='0.78', alpha=0.95),
    )


def annotate_box_column_sample_counts(
    ax,
    positions,
    box_data_list,
    fontsize=10,
    pad_data_bottom_frac=0.065,
    y_span_frac_from_bottom=0.025,
):
    """
    Place n=... just above the bottom of the y-axis range in **data coordinates**
    (avoids blended-transform bugs that can pin labels to the top).

    pad_data_bottom_frac: extend ylim downward first (room below whiskers).
    y_span_frac_from_bottom: y position = y_min + this * (y_max - y_min).
    """
    positions = list(positions)
    if pad_data_bottom_frac > 0:
        y0, y1 = ax.get_ylim()
        span = (y1 - y0) if y1 > y0 else 1.0
        ax.set_ylim(y0 - pad_data_bottom_frac * span, y1)
    y0, y1 = ax.get_ylim()
    span = (y1 - y0) if y1 > y0 else 1.0
    y_text = y0 + y_span_frac_from_bottom * span
    for i, data in enumerate(box_data_list):
        if i >= len(positions):
            break
        n = len(data)
        ax.text(
            positions[i],
            y_text,
            f'n={n}',
            ha='center',
            va='bottom',
            fontsize=fontsize,
            transform=ax.transData,
            zorder=6,
            clip_on=False,
        )
