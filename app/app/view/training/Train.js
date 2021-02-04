
Ext.define('ClassificationApp.view.training.Train',{
    extend: 'Ext.grid.Panel',
    xtype: 'train',

    requires: [
        'ClassificationApp.view.training.TrainController',
        'ClassificationApp.view.training.TrainModel'
    ],

    controller: 'training-train',
    viewModel: {
        type: 'training-train'
    },

    bind: {title: '{panel_title}',
    },

    store: 'PlannedTrainings',

    tools:[{
        type:'refresh',
        bind: {tooltip: '{tooltip_refresh}'},
        callback: 'refresh',
    },
    {
        type:'plus',
        bind: {tooltip: '{tooltip_new_planned_training}'},
        callback: 'form_new_planned_training',
    }],

    columns: [
        {
            xtype:'actioncolumn',
            width:50,
            items: [{
                iconCls: 'x-fa fa-minus-circle',
                tooltip: 'Vymazání plánovaného trénování',
                handler: 'form_delete_planned_training',
            },{
                iconCls: 'x-fa fa-edit',
                tooltip: 'Změna plánovaného trénování',
                handler: 'form_edit_planned_training',
            }]
        },
        { 
            bind: {text: '{table_data}'},  
            dataIndex: 'data',
            flex: 1
        },
        { 
            bind: {text: '{table_note}'}, 
            dataIndex: 'note', 
            flex: 1 
        },
        { 
            bind: {text: '{table_date}'}, 
            dataIndex: 'date', 
            flex: 1 
        },
        { 
            bind: {text: '{table_email}'}, 
            dataIndex: 'email', 
            flex: 1 
        },
        { 
            bind: {text: '{table_status}'}, 
            dataIndex: 'status', 
            flex: 1 
        },
        { 
            bind: {text: '{table_error}'}, 
            dataIndex: 'error', 
            flex: 1 
        },
        { 
            bind: {text: '{talbe_model}'}, 
            dataIndex: 'model', 
            flex: 1 
        },
    ],
});
