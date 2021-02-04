Ext.define('ClassificationApp.view.classification.ClassifyController', {
    extend: 'Ext.app.ViewController',
    alias: 'controller.classify',

    refresh: function(panel, tool, event) {
        ClassificationApp.getApplication().getStore('PlannedClassifications').load({
            callback: function (r, options, success) {
                if (success === true) {
                    
                }
            }
        });

        
    },

    form_new_planned_classification: function(panel, tool, event) {
        var myForm = new Ext.form.Panel({
            width: 500,
            height: 400,
            bodyPadding: 5,
            title: "Vytvoření nové plánované klasifikace",
            layout: 'anchor',
            defaults: {
                anchor: '100%'
            },
            floating: true,
            closable : true,
            defaultType: 'textfield',
            items: [
                {
                    fieldLabel: 'Data',
                    xtype: 'combobox',
                    store: 'ClassificationData',
                    queryMode: 'local',
                    displayField: 'data',
                    valueField: 'data',
                    name: 'data',
                    allowBlank: false
                },
                {
                    fieldLabel: 'Model',
                    xtype: 'combobox',
                    store: 'Models',
                    queryMode: 'local',
                    displayField: 'model',
                    valueField: 'model',
                    name: 'model',
                    allowBlank: false
                },
                {
                    fieldLabel: 'Poznámka',
                    name: 'note',
                },
                {
                    xtype: 'datefield',
                    fieldLabel: 'Dátum spuštění',
                    format: "Y-m-d",
                    name: 'date',
                    allowBlank: false
                },
                {
                    xtype: 'timefield',
                    fieldLabel: 'Čas spuštění',
                    format: "H:i",
                    name: 'time',
                    allowBlank: false
                },
                {
                    fieldLabel: 'E-mail',
                    name: 'email',
                },
            ],
            buttons: [
                { text: 'Zrušit',
                handler: function() {
                    myForm.close();
                } },
                { 
                    text: 'Vytvořit',
                    formBind: true,
                    disabled: true,
                    handler: function() {

                        var newData = myForm.getForm().findField('data').getSubmitValue();
                        var newModel = myForm.getForm().findField('model').getSubmitValue();
                        var newNote = myForm.getForm().findField('note').getSubmitValue();
                        var newDate = myForm.getForm().findField('date').getSubmitValue();
                        var newTime = myForm.getForm().findField('time').getSubmitValue();
                        var newEmail = myForm.getForm().findField('email').getSubmitValue();
                        newDate = newDate + ' ' + newTime;
                        console.log(newDate);
                        var store = panel.getStore();
                        var new_planned_classification = Ext.create('ClassificationApp.model.PlannedClassification', {
                            data: newData, model: newModel, note: newNote, date: newDate, email: newEmail
                        });
                        store.add(new_planned_classification);
                        store.sync();
                        myForm.close();
                }}
              ]
        });
        myForm.show();
    },

    form_delete_planned_classification: function(grid, rowIndex, colIndex) {
        var rec = grid.getStore().getAt(rowIndex);
        Ext.Msg.show({
            title: "Vymazání plánované klasifikace",
            message: "Opravdu chcete smazat plánovanou klasifikaci? Pokud si přejete plánovanou klasifikaci vymazat stiskněte Ano, jinak stiskněte Ne. Provedené změny nelze vrátit",
            buttons: Ext.Msg.YESNO,
            buttonText: {
                yes: 'Ano',
                no: 'Ne'
            },
            icon: Ext.Msg.QUESTION,
            fn: function(btn) {
                if (btn === 'yes') {
                    grid.getStore().remove([rowIndex]);
                    grid.getStore().sync();
                } else {
                    // console.log('Cancel pressed');
                }
            },
        });
    },

    form_edit_planned_classification: function(grid, rowIndex, colIndex) {
        var plannedClassification = grid.getStore().getAt(rowIndex);
        var myForm = new Ext.form.Panel({
            width: 500,
            height: 400,
            bodyPadding: 5,
            title: "Úprava plánované klasifikace",
            layout: 'anchor',
            defaults: {
                anchor: '100%'
            },
            floating: true,
            closable : true,
            defaultType: 'textfield',
            items: [
                {
                    fieldLabel: 'Data',
                    xtype: 'combobox',
                    store: 'ClassificationData',
                    queryMode: 'local',
                    displayField: 'data',
                    valueField: 'data',
                    name: 'data',
                    value: plannedClassification.get('data'),
                    allowBlank: false
                },
                {
                    fieldLabel: 'Model',
                    xtype: 'combobox',
                    store: 'Models',
                    queryMode: 'local',
                    displayField: 'model',
                    valueField: 'model',
                    name: 'model',
                    value: plannedClassification.get('model'),
                    allowBlank: false
                },
                {
                    fieldLabel: 'Poznámka',
                    name: 'note',
                    value: plannedClassification.get('note'),
                },
                {
                    xtype: 'datefield',
                    fieldLabel: 'Dátum spuštění',
                    name: 'date',
                    format: 'Y-m-d',
                    value: plannedClassification.get('date'),
                    allowBlank: false
                },
                {
                    xtype: 'timefield',
                    fieldLabel: 'Čas spuštění',
                    format: "H:i",
                    name: 'time',
                    value: plannedClassification.get('date'),
                    allowBlank: false
                },
                {
                    fieldLabel: 'E-mail',
                    name: 'email',
                    value: plannedClassification.get('email'),
                },
            ],
            buttons: [
                { text: 'Zrušit',
                handler: function() {
                    myForm.close();
                } },
                { 
                    text: 'Uložit',
                    formBind: true,
                    disabled: true,
                    handler: function() {

                        var values = myForm.getForm().getValues();
                        values.date = values.date + ' ' + values.time;
                        var store = grid.getStore();
                        plannedClassification.set(values);
                        store.sync();
                        myForm.close();
                }}
              ]
        });
        myForm.show();
    },
});
