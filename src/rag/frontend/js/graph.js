// Knowledge Graph visualization using vis-network

const ENTITY_COLORS = {
    PERSON: { background: '#3B82F6', border: '#2563EB', font: '#ffffff' },
    ORGANIZATION: { background: '#10B981', border: '#059669', font: '#ffffff' },
    LOCATION: { background: '#F59E0B', border: '#D97706', font: '#ffffff' },
    TOPIC: { background: '#8B5CF6', border: '#7C3AED', font: '#ffffff' },
    EVENT: { background: '#EF4444', border: '#DC2626', font: '#ffffff' },
    Document: { background: '#6B7280', border: '#4B5563', font: '#ffffff' },
};

function renderGraph(containerId, data, onNodeClick) {
    const container = document.getElementById(containerId);
    if (!container || !data.nodes || data.nodes.length === 0) return;

    const nodes = new vis.DataSet(
        data.nodes.map(n => {
            const colors = ENTITY_COLORS[n.group] || ENTITY_COLORS.TOPIC;
            return {
                id: n.id,
                label: n.label,
                group: n.group,
                color: {
                    background: colors.background,
                    border: colors.border,
                    highlight: { background: colors.background, border: '#000' },
                },
                font: { color: colors.font, size: 12 },
                shape: n.type === 'document' ? 'box' : 'dot',
                size: n.type === 'document' ? 15 : 20,
                title: `${n.label} (${n.group})`,
                _data: n,
            };
        })
    );

    const edges = new vis.DataSet(
        data.edges.map((e, i) => ({
            id: i,
            from: e.from,
            to: e.to,
            label: e.label,
            font: { size: 9, color: '#999' },
            color: { color: '#ccc', highlight: '#999' },
            arrows: { to: { enabled: true, scaleFactor: 0.5 } },
        }))
    );

    const options = {
        physics: {
            solver: 'forceAtlas2Based',
            forceAtlas2Based: {
                gravitationalConstant: -40,
                centralGravity: 0.005,
                springLength: 150,
                springConstant: 0.08,
            },
            stabilization: { iterations: 100 },
        },
        interaction: {
            hover: true,
            tooltipDelay: 200,
        },
        layout: {
            improvedLayout: true,
        },
    };

    const network = new vis.Network(container, { nodes, edges }, options);

    network.on('doubleClick', (params) => {
        if (params.nodes.length > 0) {
            const nodeId = params.nodes[0];
            const node = nodes.get(nodeId);
            if (node._data && node._data.type === 'entity' && onNodeClick) {
                onNodeClick(node._data.label);
            } else if (node._data && node._data.type === 'document' && node._data.doc_id) {
                window.open('/documents/' + node._data.doc_id, '_blank');
            }
        }
    });

    return network;
}
