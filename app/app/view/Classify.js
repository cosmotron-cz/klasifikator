
Ext.define('ClassificationApp.view.Classify',{
    extend: 'Ext.form.Panel',
    xtype: 'classify',

    requires: [
        'ClassificationApp.view.ClassifyController',
        'ClassificationApp.view.ClassifyModel'
    ],

    controller: 'classify',
    viewModel: {
        type: 'classify'
    },

    bodyPadding: 5,

    // Fields will be arranged vertically, stretched to full width
    layout: 'anchor',
    defaults: {
        anchor: '100%'
    },
    title: 'Klasifikace',

    // The fields
    defaultType: 'textfield',
    items: [{
        fieldLabel: 'Adresář',
        name: 'directory',
        allowBlank: false
    },{
        fieldLabel: 'Export',
        name: 'export_to',
        allowBlank: false
    }],

    // listeners : {
    //     order : 'before',
    //     beforesubmit : function(form, values) {
            
    //         return false;
    //     }
    // },

    // Reset and Submit buttons
    buttons: [{
        text: 'Reset',
        handler: function() {
            this.up('form').getForm().reset();
        }
    }, {
        text: 'Klasifikuj',
        formBind: true, //only enabled once the form is valid
        disabled: true,
        handler: function() {
            var url = window.location.hostname;
            url = 'http://' + url + ':5001/run_classification';
            console.log(url);
            var form = this.up('form').getForm();
            if (form.isValid()) {
                Ext.data.JsonP.request({
                    url: url,
                    //callbackKey: 'callback',
                    params: {
                        directory: form.getValues().directory,
                        export_to: form.getValues().export_to
                    },
                    success: function(result, request) {
                        //Ext.Msg.alert('Success', 'Klasifikace se spustila');
                    },
                    failure: function(result, request) {
                        //Ext.Msg.alert('Failed', 'Chyba klasifikace');
                    }  
                });
            }
        }
    }],

});
