
Ext.define('ClassificationApp.view.classification.Classify',{
    extend: 'Ext.grid.Panel',
    xtype: 'classify',

    requires: [
        'ClassificationApp.view.classification.ClassifyController',
        'ClassificationApp.view.classification.ClassifyModel',
        'ClassificationApp.store.ClassificationData',
        'ClassificationApp.store.Models',
        'ClassificationApp.store.PlannedClassifications'
    ],

    controller: 'classify',
    viewModel: {
        type: 'classify'
    },

    bind: {title: '{panel_title}',
    },

    store: 'PlannedClassifications',

    tools:[{
        type:'refresh',
        bind: {tooltip: '{tooltip_refresh}'},
        callback: 'refresh',
    },
    {
        type:'plus',
        bind: {tooltip: '{tooltip_new_planned_classification}'},
        callback: 'form_new_planned_classification',
    }],

    columns: [
        {
            xtype:'actioncolumn',
            width:50,
            items: [{
                iconCls: 'x-fa fa-minus-circle',
                tooltip: 'Vymazání plánované klasifikace',
                handler: 'form_delete_planned_classification',
            },{
                iconCls: 'x-fa fa-edit',
                tooltip: 'Změna plánované klasifikace',
                handler: 'form_edit_planned_classification',
            }]
        },
        { 
            bind: {text: '{table_data}'},  
            dataIndex: 'data',
            flex: 1
        },
        { 
            bind: {text: '{talbe_model}'}, 
            dataIndex: 'model', 
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
            bind: {text: '{table_export}'}, 
            dataIndex: 'export', 
            flex: 1 
        }
    ],

});
